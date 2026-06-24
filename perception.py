import cv2
from ultralytics import YOLO
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import threading
from collections import deque, Counter, defaultdict

class PerceptionModule:
    def __init__(self, hand_model_path="hand_landmarker.task", object_model_path="best_3.pt"):
        self.hand_model_path = hand_model_path
        self.object_model_path = object_model_path

        # --- YOLO ---
        self.object_model = YOLO(self.object_model_path)
        self.class_names = self.object_model.names
        # Model state is not thread-safe; robot worker thread also calls it.
        self._yolo_lock = threading.Lock()

        # Object tracking state
        self.id_counter = defaultdict(int)
        self.confirmed_objects = set()
        self.persistent_objects = {}
        self.last_seen_frame = {}
        self.latest_objects = []
        self.MIN_FRAMES_CONFIRM = 8
        self.YOLO_FREQ = 3          # run detection every N frames
        self.MAX_ABSENT_FRAMES = 30 # frames before pruning a missing object

        # --- MediaPipe ---
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

        self.base_options = python.BaseOptions(model_asset_path=hand_model_path)
        self.options = vision.HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self.save_hand_result
        )

        self.landmarker = vision.HandLandmarker.create_from_options(self.options)
        self.latest_hand_result = None
        
        # --- Thresholds & smoothing ---
        self.PINCH_THRESHOLD = 0.2          # normalized 3-D thumb-index distance
        self.ASSEMBLY_WRIST_THRESHOLD = 0.7 # normalized wrist-wrist distance
        self.HAND_OBJECT_PROXIMITY = 50     # px
        self.WORKING_AREA_THRESHOLD = 200   # px
        self.SLIDING_WINDOW_MS = 300        # ms
        self.global_state_history = deque()
        self.semantic_action_history = deque()

        # Object-count delta: appends "recent: X entered; Y left" to semantic action
        # so the LLM can disambiguate steps with identical gesture descriptions.
        self.INVENTORY_DELTA_WINDOW_MS = 1500
        self._inventory_history = deque()  # (timestamp_ms, frozenset(class_names))
        # Sticky: accumulates events until consume_delta() is called after LLM dispatch.
        # Prevents "spring entered" from clearing before inference (latency 1–6 s).
        self._pending_delta_parts: list[str] = []

        self.frame_count = 0

    def save_hand_result(self, result, output_image, timestamp_ms):
        """MediaPipe async callback — stores the latest hand landmark result."""
        self.latest_hand_result = result

    def extract_relevant_keypoints(self, detection_result, width, height):
        """Return pixel + normalized coords for wrist, thumb tip, index tip per hand.
        Returns: dict {'Left': {'wrist': (x,y), ...}, 'Right': ...}
        """
        keypoints = {}
        if detection_result and detection_result.hand_landmarks:
            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                if i < len(detection_result.handedness):
                    handedness = detection_result.handedness[i][0].category_name
                    # Landmark indices: 0=Wrist, 4=Thumb Tip, 8=Index Tip
                    kp = {
                        "wrist": (int(hand_landmarks[0].x * width), int(hand_landmarks[0].y * height)),
                        "thumb": (int(hand_landmarks[4].x * width), int(hand_landmarks[4].y * height)),
                        "index": (int(hand_landmarks[8].x * width), int(hand_landmarks[8].y * height)),
                        "thumb_norm": (hand_landmarks[4].x, hand_landmarks[4].y, hand_landmarks[4].z),
                        "index_norm": (hand_landmarks[8].x, hand_landmarks[8].y, hand_landmarks[8].z),
                        "wrist_norm": (hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z)
                    }
                    keypoints[handedness] = kp
        return keypoints

    def elaborate_state(self, keypoints):
        """Classify hand state as 'pinch', 'assembly', or 'nothing'.

        'assembly' requires both hands pinching AND wrists close together,
        capturing the two-handed gesture of joining two components.
        """
        states = {}
        pinching_hands = []

        for hand, kp in keypoints.items():
            t, i = kp['thumb_norm'], kp['index_norm']
            dist = math.sqrt((t[0]-i[0])**2 + (t[1]-i[1])**2 + (t[2]-i[2])**2)
            if dist < self.PINCH_THRESHOLD:
                states[hand] = "pinch"
                pinching_hands.append(hand)
            else:
                states[hand] = "nothing"

        final_state = "nothing"
        if len(pinching_hands) == 2:
            l_wrist = keypoints['Left']['wrist_norm']  if 'Left'  in keypoints else None
            r_wrist = keypoints['Right']['wrist_norm'] if 'Right' in keypoints else None
            if l_wrist and r_wrist:
                wrist_dist = math.sqrt((l_wrist[0]-r_wrist[0])**2 + (l_wrist[1]-r_wrist[1])**2)
                final_state = "assembly" if wrist_dist < self.ASSEMBLY_WRIST_THRESHOLD else "pinch"
        elif len(pinching_hands) > 0:
            final_state = "pinch"

        return final_state, states

    def process_yolo(self, frame):
        """Run YOLO tracking, update confirmed-object registry, return latest objects."""
        if self.frame_count % self.YOLO_FREQ == 0:
            with self._yolo_lock:
                results = self.object_model.track(frame, show=False, persist=True, verbose=False)

            if results and results[0].boxes.id is not None:
                boxes      = results[0].boxes.xywh.cpu().numpy()
                box_coords = results[0].boxes.xyxy.cpu().numpy()
                classes    = results[0].boxes.cls.cpu().numpy()
                ids        = results[0].boxes.id.cpu().numpy()
                masks      = results[0].masks

                present_confirmed_ids = set()
                detections = []
                for i, box in enumerate(boxes):
                    track_id   = int(ids[i])
                    class_name = self.class_names[int(classes[i])]
                    if (class_name, track_id) in self.confirmed_objects:
                        present_confirmed_ids.add((class_name, track_id))
                    detections.append({
                        'class': class_name, 'id': track_id, 'box': box,
                        'coords': box_coords[i],
                        'mask': masks.xy[i] if masks is not None else None
                    })

                for det in detections:
                    class_name = det['class']
                    track_id   = det['id']
                    cx, cy, *_ = det['box']
                    object_key = (class_name, track_id)

                    # ID recovery: transfer identity if tracker ID was lost (e.g. occlusion).
                    if object_key not in self.confirmed_objects:
                        candidates = [k for k in self.confirmed_objects
                                      if k[0] == class_name and k not in present_confirmed_ids]
                        if candidates:
                            old_key = candidates[0]
                            self.confirmed_objects.remove(old_key)
                            self.confirmed_objects.add(object_key)
                            self.persistent_objects.pop(old_key, None)
                            self.last_seen_frame.pop(old_key, None)
                            self.id_counter[object_key] = self.id_counter.pop(old_key, 0)

                    self.last_seen_frame[object_key] = self.frame_count
                    self.id_counter[object_key] += 1
                    if self.id_counter[object_key] >= self.MIN_FRAMES_CONFIRM:
                        self.confirmed_objects.add(object_key)

                    self.persistent_objects[object_key] = {
                        'class': class_name, 'id': track_id,
                        'center': (cx, cy),
                        'bbox': det['coords'],  # x1, y1, x2, y2
                        'mask': det['mask']
                    }

            # Prune objects absent for too long
            stale = {k for k in self.confirmed_objects
                     if self.frame_count - self.last_seen_frame.get(k, self.frame_count) > self.MAX_ABSENT_FRAMES}
            for key in stale:
                self.confirmed_objects.discard(key)
                self.persistent_objects.pop(key, None)
                self.last_seen_frame.pop(key, None)
                self.id_counter.pop(key, None)

            self.latest_objects = [self.persistent_objects[k] for k in self.confirmed_objects
                                    if k in self.persistent_objects]
        
        self.frame_count += 1
        return self.latest_objects

    def get_confirmed_classes(self) -> set:
        """Return class names of objects currently confirmed present in the scene.
        Uses the perception-camera YOLO tracker (3-frame confirmation threshold)."""
        return {class_name for class_name, _ in self.confirmed_objects}

    def detect_objects_in_frame(self, frame):
        """Stateless single-frame detection (no tracking). Used for the robot camera."""
        with self._yolo_lock:
            results = self.object_model(frame, show=False, verbose=False)
        detections = []
        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                cls_id = int(classes[i])
                detections.append({
                    'class': self.class_names[cls_id],
                    'coords': box,
                    'confidence': float(confs[i]),
                })

        return detections

    def detect_objects_in_frames(self, frames, min_frame_count=3):
        """Detect objects across multiple frames and return aggregated results.

        Matches detections across frames by class + center proximity; returns
        only objects seen in at least min_frame_count frames, with mean bbox
        and max confidence. Used by the robot worker for more robust detection.
        """
        MATCH_DIST = 100  # px — max center distance to match the same object across frames
        tracks = []

        for frame in frames:
            with self._yolo_lock:
                results = self.object_model(frame, show=False, verbose=False)
            if not (results and results[0].boxes):
                continue
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs   = results[0].boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                cls_id     = int(classes[i])
                class_name = self.class_names[cls_id]
                conf       = float(confs[i])
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2

                best_track = None  # nearest track of same class within MATCH_DIST
                best_dist  = float('inf')
                for track in tracks:
                    if track['class'] != class_name:
                        continue
                    mean_box = np.mean(track['coords_list'], axis=0)
                    tcx = (mean_box[0] + mean_box[2]) / 2
                    tcy = (mean_box[1] + mean_box[3]) / 2
                    dist = math.sqrt((cx - tcx) ** 2 + (cy - tcy) ** 2)
                    if dist < best_dist:
                        best_dist  = dist
                        best_track = track

                if best_track is not None and best_dist < MATCH_DIST:
                    best_track['coords_list'].append(box)
                    best_track['confs'].append(conf)
                else:
                    tracks.append({'class': class_name, 'coords_list': [box], 'confs': [conf]})

        aggregated = []
        for track in tracks:
            if len(track['coords_list']) >= min_frame_count:
                aggregated.append({
                    'class':  track['class'],
                    'coords': np.mean(track['coords_list'], axis=0),
                    'confidence': max(track['confs']),
                })

        return aggregated

    def calculate_semantic_action(self, global_state, hand_states, keypoints, objects):
        """Map hand state + object proximity to a semantic action string."""
        action = "nothing"
        involved_objects = []

        if not objects:
            return action, involved_objects

        def get_closest_object(point_px):
            min_dist = float('inf')
            closest_obj = None
            for obj in objects:
                ocx, ocy = obj['center']
                dist = math.sqrt((point_px[0]-ocx)**2 + (point_px[1]-ocy)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_obj = obj
            return closest_obj, min_dist

        if global_state == "pinch":
            pinched_objects_info = []
            for hand, state in hand_states.items():
                if state == "pinch":
                    tx, ty = keypoints[hand]['thumb']
                    ix, iy = keypoints[hand]['index']
                    mx, my = (tx+ix)/2, (ty+iy)/2  # thumb-index midpoint
                    obj, dist = get_closest_object((mx, my))
                    if obj and dist < self.HAND_OBJECT_PROXIMITY:
                        pinched_objects_info.append(obj)

            # Deduplicate by track ID (both hands may touch the same object)
            unique_pinched_objects = []
            seen_ids = set()
            for obj in pinched_objects_info:
                if obj['id'] not in seen_ids:
                    unique_pinched_objects.append(obj)
                    seen_ids.add(obj['id'])

            if len(unique_pinched_objects) == 1:
                action = f"pinch {unique_pinched_objects[0]['class']}"
                involved_objects = [unique_pinched_objects[0]['class']]
            elif len(unique_pinched_objects) >= 2:
                class_names = sorted([p['class'] for p in unique_pinched_objects])
                action = f"pinch {', '.join(class_names)}"
                involved_objects = sorted(list(set(class_names)))

        elif global_state == "assembly":
            if 'Left' in keypoints and 'Right' in keypoints:
                lx, ly = keypoints['Left']['wrist']
                rx, ry = keypoints['Right']['wrist']
                cx, cy = (lx+rx)/2, (ly+ry)/2  # midpoint between wrists
                nearby_objects = []
                for obj in objects:
                    ocx, ocy = obj['center']
                    dist = math.sqrt((cx-ocx)**2 + (cy-ocy)**2)
                    if dist < self.HAND_OBJECT_PROXIMITY * 4:  # wider radius for assembly gesture
                        nearby_objects.append(obj['class'])
                
                if len(nearby_objects) >= 1:
                    action = f"assembly with {', '.join(set(nearby_objects))}"
                    involved_objects = list(set(nearby_objects))
                else:
                    action = "assembly (no object detected)"

        return action, involved_objects

    def _snapshot_inventory(self, timestamp_ms):
        """Append a confirmed-class snapshot on change; prune entries older than 2× window."""
        classes = frozenset(cls for cls, _ in self.confirmed_objects)
        if not self._inventory_history or self._inventory_history[-1][1] != classes:
            self._inventory_history.append((timestamp_ms, classes))
        cutoff = timestamp_ms - 2 * self.INVENTORY_DELTA_WINDOW_MS
        while len(self._inventory_history) > 1 and self._inventory_history[0][0] < cutoff:
            self._inventory_history.popleft()

    def _inventory_delta(self, timestamp_ms):
        """Return (entered, left) comparing now vs the snapshot from ~window_ms ago.
        Returns empty lists if no reference snapshot exists yet."""
        target = timestamp_ms - self.INVENTORY_DELTA_WINDOW_MS
        reference = None
        for t, classes in self._inventory_history:
            if t <= target:
                reference = classes
            else:
                break
        if reference is None:
            return [], []
        current = frozenset(cls for cls, _ in self.confirmed_objects)
        entered = sorted(current - reference)
        left    = sorted(reference - current)
        return entered, left

    def consume_delta(self):
        """Clear pending delta events. Called after LLM dispatch, not on cache-replay."""
        self._pending_delta_parts.clear()

    def process_frame(self, frame):
        height, width, _ = frame.shape
        timestamp_ms = int(time.time() * 1000)

        # 1. Hand detection (async — result arrives in save_hand_result callback)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self.landmarker.detect_async(mp_image, timestamp_ms)

        # 2. YOLO object tracking
        self.process_yolo(frame)

        # 3. Instantaneous state (latest_objects includes last-known position of occluded objects).
        keypoints = self.extract_relevant_keypoints(self.latest_hand_result, width, height)
        instantaneous_global_state, instantaneous_hand_states = self.elaborate_state(keypoints)
        instantaneous_semantic_action, instantaneous_involved_objs = self.calculate_semantic_action(
            instantaneous_global_state, instantaneous_hand_states, keypoints, self.latest_objects
        )

        # 4. Majority-vote smoothing over 300 ms sliding window.
        self.global_state_history.append((timestamp_ms, instantaneous_global_state))
        self.semantic_action_history.append((timestamp_ms, instantaneous_semantic_action, instantaneous_involved_objs))

        while self.global_state_history and (timestamp_ms - self.global_state_history[0][0]) > self.SLIDING_WINDOW_MS:
            self.global_state_history.popleft()
        while self.semantic_action_history and (timestamp_ms - self.semantic_action_history[0][0]) > self.SLIDING_WINDOW_MS:
            self.semantic_action_history.popleft()

        smoothed_global_state    = "nothing"
        smoothed_semantic_action = "nothing"
        smoothed_involved_objs   = []

        if self.global_state_history:
            states = [s for _, s in self.global_state_history]
            smoothed_global_state = Counter(states).most_common(1)[0][0]

        if self.semantic_action_history:
            actions = [a for _, a, _ in self.semantic_action_history]
            most_common_action_str = Counter(actions).most_common(1)[0][0]
            smoothed_semantic_action = most_common_action_str
            for _, action_str, objs in reversed(self.semantic_action_history):
                if action_str == most_common_action_str:
                    smoothed_involved_objs = objs
                    break

        # 4b. Sticky delta — "recent: X entered" disambiguates steps with identical actions.
        self._snapshot_inventory(timestamp_ms)
        entered, left = self._inventory_delta(timestamp_ms)
        new_parts = []
        if entered:
            new_parts.append(f"{', '.join(entered)} entered")
        if left:
            new_parts.append(f"{', '.join(left)} left")
        for part in new_parts:
            if part not in self._pending_delta_parts:
                self._pending_delta_parts.append(part)
        if len(self._pending_delta_parts) > 4:
            self._pending_delta_parts = self._pending_delta_parts[-4:]
        augmented_semantic_action = (
            f"{smoothed_semantic_action} | recent: {'; '.join(self._pending_delta_parts)}"
            if self._pending_delta_parts else smoothed_semantic_action
        )

        # 5. Visualization
        annotated_frame = frame.copy()

        for obj in self.latest_objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            # Color based on class hash
            color = (int(hash(obj['class']) % 255), 255, 0)
            
            # Draw Mask
            if obj['mask'] is not None and len(obj['mask']) > 0:
                mask_pts = np.array(obj['mask'], dtype=np.int32)
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [mask_pts], color)
                cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)
                cv2.polylines(annotated_frame, [mask_pts], True, color, 2)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, obj['class'], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.latest_hand_result and self.latest_hand_result.hand_landmarks:
            for hand_landmarks in self.latest_hand_result.hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

        # Split delta suffix onto its own HUD line to avoid clipping.
        if " | recent:" in augmented_semantic_action:
            base, delta = augmented_semantic_action.split(" | recent:", 1)
            hud_lines = [
                (f"Action: {base}",          (255, 255, 255)),
                (f"  recent: {delta.strip()}", (0, 220, 255)),
                (f"Hands: {instantaneous_hand_states}", (200, 200, 200)),
            ]
        else:
            hud_lines = [
                (f"Action: {augmented_semantic_action}", (255, 255, 255)),
                (f"Hands: {instantaneous_hand_states}",  (200, 200, 200)),
            ]

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness  = 2
        line_h     = 22
        pad        = 4

        # Semi-transparent panel behind HUD lines.
        panel_w = max(
            cv2.getTextSize(t, font, font_scale, thickness)[0][0]
            for t, _ in hud_lines
        ) + pad * 2
        panel_h = len(hud_lines) * line_h + pad * 2
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (6, 8), (6 + panel_w, 8 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, annotated_frame, 0.55, 0, annotated_frame)

        for i, (line, color) in enumerate(hud_lines):
            y = 8 + pad + (i + 1) * line_h - 4
            cv2.putText(annotated_frame, line, (10, y),
                        font, font_scale, color, thickness, cv2.LINE_AA)

        # 6. is_assembly from pre-delta action — delta suffix must not affect LLM/VLM routing.
        is_assembly = 1 if "assembly" in smoothed_semantic_action and len(smoothed_involved_objs) >= 2 else 0

        log_entry = {
            "timestamp": timestamp_ms,
            "confirmed_objects": list({cls for cls, _ in self.confirmed_objects}),
            "current_action": augmented_semantic_action,   # with delta → sent to LLM
            "smoothed_action": smoothed_semantic_action,  # pre-delta → used for Counter/dedup
            "involved_objects": smoothed_involved_objs,
            "hand_state": smoothed_global_state,
            "instantaneous_hand_state": instantaneous_global_state,
            "is_assembly": is_assembly,
            "inventory_entered": entered,
            "inventory_left": left,
        }

        return log_entry, annotated_frame

    def close(self):
        self.landmarker.close()