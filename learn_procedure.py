"""
LEARN PROCEDURE — Approach 2: Autonomous Builder with Post-Hoc VLM Cleanup

Watches an operator perform an assembly task once, records raw observations
via Gemini VLM, then uses Gemini again with the captured frames to clean and
deduplicate the observations into a structured memory.json.

Usage:
  python learn_procedure.py
  python learn_procedure.py --serial <camera_serial>
  python learn_procedure.py --webcam [--webcam-index 1]
  python learn_procedure.py --output learned_memory.json
  python learn_procedure.py --displacement-threshold 40
  python learn_procedure.py --cooldown-frames 90

Required: GEMINI_API_KEY environment variable.
"""

import cv2
import numpy as np
import argparse

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False

from perception import PerceptionModule
from llm import LLM_planner
from learn_utils import (
    WebcamPipeline,
    capture_frames, extract_action_frames,
    checkpoint_buffer, load_checkpoint, delete_checkpoint,
    checkpoint_frames, load_frame_checkpoint, delete_frame_checkpoint,
    save_memory, phase3_validate,
)

CHECKPOINT_FILE      = "learn_procedure_checkpoint.json"
FRAME_CHECKPOINT_DIR = "learn_frames"


class TriggerDetector:
    """
    Detects when an assembly action has completed based on two conditions:
      1. Hand state downgrade: active (pinch/assembly) → inactive (nothing/pinch)
      2. Object displacement: a tracked object moved or disappeared from its
         pre-action position — confirms that a part was actually placed.

    State machine: IDLE → ACTIVE → CHECK_DISPLACEMENT → COOLDOWN → IDLE

    The displacement check guards against false triggers from brief, accidental
    hand contact that doesn't result in any part placement.

    Fallback: trigger fires unconditionally if the operator stays active longer
    than max_active_duration_ms (handles long or ambiguous gestures).
    """

    IDLE               = "idle"
    ACTIVE             = "active"
    CHECK_DISPLACEMENT = "check_displacement"
    COOLDOWN           = "cooldown"

    # Hierarchy used to detect state downgrades: assembly > pinch > nothing.
    STATE_LEVEL = {"nothing": 0, "pinch": 1, "assembly": 2}

    def __init__(self, displacement_threshold_px=40, min_active_duration_ms=300,
                 cooldown_frames=90, max_active_duration_ms=20000,
                 check_displacement_grace_frames=15,
                 release_confirm_frames=3):
        self.displacement_threshold_px       = displacement_threshold_px
        self.min_active_duration_ms          = min_active_duration_ms
        self.cooldown_frames                 = cooldown_frames
        self.max_active_duration_ms          = max_active_duration_ms
        self.check_displacement_grace_frames = check_displacement_grace_frames
        self.release_confirm_frames          = release_confirm_frames
        self.reset()

    def reset(self):
        self.state                       = self.IDLE
        self.active_since_ms             = None
        self.pre_action_centroids        = {}
        self.cooldown_remaining          = 0
        self.check_displacement_remaining = 0
        self.peak_level                  = 0
        self.consecutive_below           = 0

    def _snapshot_centroids(self, persistent_objects):
        return {
            key: (obj['center'][0], obj['center'][1])
            for key, obj in persistent_objects.items()
        }

    def _check_displacement(self, persistent_objects):
        """
        Return a reason string if any tracked object moved beyond the threshold
        or disappeared entirely, else None.
        """
        current = self._snapshot_centroids(persistent_objects)

        for key in self.pre_action_centroids:
            if key not in current:
                return f"object disappeared: {key}"

        for key, (old_cx, old_cy) in self.pre_action_centroids.items():
            if key in current:
                new_cx, new_cy = current[key]
                dist = np.sqrt((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2)
                if dist > self.displacement_threshold_px:
                    return f"object moved: {key}  ({dist:.0f}px > {self.displacement_threshold_px}px)"

        return None

    def update(self, log_entry, persistent_objects):
        """
        Process one frame of perception data. Returns True if a trigger fires.

        Uses the INSTANTANEOUS hand state (not smoothed) to detect state
        downgrades: assembly → pinch (component released) or pinch → nothing
        (fully disengaged).  Either downgrade held for release_confirm_frames
        consecutive frames transitions to CHECK_DISPLACEMENT.
        """
        hand_state    = log_entry.get("instantaneous_hand_state", log_entry["hand_state"])
        timestamp_ms  = log_entry["timestamp"]
        current_level = self.STATE_LEVEL.get(hand_state, 0)

        if self.state == self.COOLDOWN:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining <= 0:
                self.state = self.IDLE
            return False

        if self.state == self.IDLE:
            if current_level >= 1:
                self.state            = self.ACTIVE
                self.active_since_ms  = timestamp_ms
                self.peak_level       = current_level
                self.consecutive_below = 0
                self.pre_action_centroids = self._snapshot_centroids(persistent_objects)
                print(f"  [Trigger] IDLE → ACTIVE  (hand={hand_state}, "
                      f"objects_snapshot={list(self.pre_action_centroids.keys())})")
            return False

        if self.state == self.ACTIVE:
            duration = timestamp_ms - (self.active_since_ms or timestamp_ms)

            if current_level >= self.peak_level:
                if current_level > self.peak_level:
                    print(f"  [Trigger] ACTIVE peak updated: "
                          f"{list(self.STATE_LEVEL.keys())[self.peak_level]} → {hand_state}")
                self.peak_level        = current_level
                self.consecutive_below = 0
            else:
                self.consecutive_below += 1

            if duration >= self.max_active_duration_ms:
                print(f"  [Trigger] ACTIVE → COOLDOWN  (max duration {duration/1000:.1f}s reached)")
                self.state              = self.COOLDOWN
                self.cooldown_remaining = self.cooldown_frames
                return True

            if self.consecutive_below >= self.release_confirm_frames:
                self.consecutive_below = 0
                if duration < self.min_active_duration_ms:
                    print(f"  [Trigger] ACTIVE → IDLE  (too brief: {duration:.0f}ms < {self.min_active_duration_ms}ms)")
                    self.state = self.IDLE
                    return False
                print(f"  [Trigger] ACTIVE → CHECK_DISPLACEMENT  "
                      f"(hand dropped to '{hand_state}' after {duration:.0f}ms, "
                      f"peak was '{list(self.STATE_LEVEL.keys())[self.peak_level]}')")
                self.state                        = self.CHECK_DISPLACEMENT
                self.check_displacement_remaining = self.check_displacement_grace_frames
                return False

            return False

        if self.state == self.CHECK_DISPLACEMENT:
            reason = self._check_displacement(persistent_objects)
            if reason:
                print(f"  [Trigger] CHECK_DISPLACEMENT → COOLDOWN  ({reason})")
                self.state              = self.COOLDOWN
                self.cooldown_remaining = self.cooldown_frames
                return True
            else:
                self.check_displacement_remaining -= 1
                if self.check_displacement_remaining <= 0:
                    print("  [Trigger] CHECK_DISPLACEMENT → IDLE  "
                          "(no displacement detected, discarding action)")
                    self.state = self.IDLE
                return False

        return False


def phase0_inventory_scan(pipeline, perception, auto_timeout_frames=450):
    """
    Scan the initial scene to build a complete object inventory.
    Returns (inventory, reference_frame) — reference_frame is a YOLO-annotated
    image that lets infer_VLM_learn map label names to visual appearance.
    """
    print("\n" + "=" * 55)
    print("  PHASE 0 — OBJECT INVENTORY SCAN")
    print("  Place ALL assembly objects visibly on the table.")
    print("  Press SPACE to confirm, or wait for auto-scan.")
    print("=" * 55 + "\n")

    all_detected   = set()
    frames_scanned = 0
    confirmed      = False
    last_frame     = None

    while not confirmed:
        fs = pipeline.wait_for_frames()
        color = fs.get_color_frame()
        if not color:
            continue
        frame      = np.asanyarray(color.get_data())
        last_frame = frame
        _, annotated_frame = perception.process_frame(frame)

        for cls, _ in perception.confirmed_objects:
            all_detected.add(cls)

        frames_scanned += 1

        cv2.putText(annotated_frame, f"Objects detected: {len(all_detected)}",
                    (10, annotated_frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "SPACE to confirm",
                    (10, annotated_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Inventory Scan", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            confirmed = True

        if not confirmed and _HAS_MSVCRT:
            while msvcrt.kbhit():
                if msvcrt.getch() == b' ':
                    confirmed = True

        if frames_scanned >= auto_timeout_frames:
            print("  [Inventory] Auto-capture timeout reached.")
            confirmed = True

    inventory = sorted(all_detected)
    if inventory:
        print(f"  [Inventory] Captured {len(inventory)} object(s): {', '.join(inventory)}")
    else:
        print("  [Inventory] No objects detected. Proceeding without inventory.")

    reference_frame = None
    if last_frame is not None:
        results = perception.object_model(last_frame, verbose=False)
        if results and results[0].boxes:
            reference_frame = results[0].plot()
            print("  [Inventory] YOLO reference frame created.")
        else:
            print("  [Inventory] No objects detected for reference frame.")

    return inventory, reference_frame


def phase1_record(pipeline, perception, planner, trigger_detector,
                   initial_buffer=None, initial_frame_buffer=None,
                   reference_frame=None):
    """
    Observe the assembly and record raw VLM observations autonomously.
    Returns (raw_buffer, frame_buffer).
    """
    raw_buffer   = initial_buffer        if initial_buffer        else []
    frame_buffer = initial_frame_buffer  if initial_frame_buffer  else []
    step_count   = len(raw_buffer)

    # Rolling buffer of active-phase frames; before/mid/mid extracted at trigger time.
    active_frame_buffer = []
    MAX_ACTIVE_FRAMES   = 60  # ~2 s at 30 fps

    IN_ACTION_STATES = (TriggerDetector.ACTIVE, TriggerDetector.CHECK_DISPLACEMENT)

    print("\n" + "=" * 55)
    print("  PHASE 1 — AUTONOMOUS RECORDING")
    print("  Perform the assembly. Press Q when done.")
    if step_count > 0:
        print(f"  Resuming from checkpoint ({step_count} steps already recorded)")
    print("=" * 55 + "\n")

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        log_entry, annotated_frame = perception.process_frame(frame)

        prev_state = trigger_detector.state
        triggered  = trigger_detector.update(log_entry, perception.persistent_objects)

        if trigger_detector.state in IN_ACTION_STATES:
            active_frame_buffer.append(frame)
            if len(active_frame_buffer) > MAX_ACTIVE_FRAMES:
                active_frame_buffer.pop(0)
        elif prev_state in IN_ACTION_STATES and not triggered:
            active_frame_buffer.clear()

        if triggered:
            if not perception.confirmed_objects:
                print("  [Skip] Trigger fired but no confirmed objects in scene.")
                active_frame_buffer.clear()
            else:
                print(f"\n  [Trigger] Action detected ({len(active_frame_buffer)} active frames).")

                start_frame, mid1, mid2 = extract_action_frames(active_frame_buffer)
                active_frame_buffer.clear()

                end_frames = capture_frames(pipeline, n_frames=1)
                end_frame  = end_frames[0] if end_frames else frame

                learn_frames   = [start_frame, mid1, mid2, end_frame]  # before/mounting/mounting/after
                cleanup_frames = [mid1, mid2]  # mid-action frames for VLM_cleanup cross-referencing

                print("\n  [VLM] Analyzing action...")
                vlm_result = planner.infer_VLM_learn(learn_frames, reference_frame=reference_frame)

                if vlm_result:
                    step_count += 1
                    observation = {
                        "timestamp_ms":     log_entry["timestamp"],
                        "description":      vlm_result.get("description", ""),
                        "objects_required": vlm_result.get("objects_required", []),
                        "raw_step_index":   step_count,
                    }
                    raw_buffer.append(observation)
                    frame_buffer.append({"step_index": step_count, "frames": cleanup_frames})

                    checkpoint_buffer(raw_buffer, CHECKPOINT_FILE)
                    checkpoint_frames(step_count, cleanup_frames, FRAME_CHECKPOINT_DIR)

                    print(f"  [Step {step_count}] {observation['description']}")
                    print(f"              Objects: {', '.join(observation['objects_required'])} \n")
                else:
                    print("  [VLM] No result returned. Skipping.")

        trigger_colors = {
            "idle": (180, 180, 180), "active": (0, 255, 0),
            "check_displacement": (0, 255, 255), "cooldown": (0, 165, 255),
        }
        t_color = trigger_colors.get(trigger_detector.state, (200, 200, 200))
        cv2.putText(annotated_frame, f"Steps: {step_count}",
                    (10, annotated_frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Trigger: {trigger_detector.state.upper()}",
                    (10, annotated_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, t_color, 1)
        cv2.putText(annotated_frame, "Q to stop",
                    (annotated_frame.shape[1] - 130, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Learn — Perception", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        quit_requested = key in (ord('q'), ord('Q'))

        if not quit_requested and _HAS_MSVCRT:
            while msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch.lower() == b'q':
                    quit_requested = True

        if quit_requested:
            break

    cv2.destroyAllWindows()
    return raw_buffer, frame_buffer


def main():
    parser = argparse.ArgumentParser(description="Learn assembly procedure by demonstration")
    parser.add_argument("--serial", type=str, default=None,
                        help="RealSense camera serial number (uses first available if omitted)")
    parser.add_argument("--webcam", action="store_true",
                        help="Use a webcam instead of a RealSense camera")
    parser.add_argument("--webcam-index", type=int, default=0,
                        help="Webcam device index (default: 0); only used with --webcam")
    parser.add_argument("--output", type=str, default="learned_memory.json",
                        help="Output file path (default: learned_memory.json)")
    parser.add_argument("--displacement-threshold", type=int, default=40,
                        help="Pixel displacement threshold for trigger (default: 40)")
    parser.add_argument("--cooldown-frames", type=int, default=90,
                        help="Frames to ignore after a trigger (default: 90)")
    parser.add_argument("--max-active-duration", type=int, default=20,
                        help="Seconds before a trigger fires unconditionally "
                             "while hand is still active (default: 20)")
    args = parser.parse_args()

    if args.webcam:
        print(f"\nStarting webcam (index {args.webcam_index})...")
        pipeline = WebcamPipeline(index=args.webcam_index)
    else:
        if not _RS_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not installed. Use --webcam to use a webcam instead.")
        pipeline = rs.pipeline()
        config = rs.config()
        if args.serial:
            config.enable_device(args.serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        print("\nStarting RealSense camera...")
        pipeline.start(config)

    # Learning phase only needs Gemini — local Qwen is for assembly inference only.
    print("Loading LLM_planner (Gemini only)...")
    planner = LLM_planner(load_local_model=False)

    initial_buffer = None
    initial_frame_buffer = None
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    if checkpoint:
        print(f"\n  Found checkpoint with {len(checkpoint)} raw observations from a previous session.")
        while True:
            choice = input("  [C] Continue from checkpoint    [D] Discard and start fresh: ").strip().upper()
            if choice == 'C':
                initial_buffer       = checkpoint
                initial_frame_buffer = load_frame_checkpoint(checkpoint, FRAME_CHECKPOINT_DIR)
                print(f"  Loaded {len(initial_frame_buffer)} frame sets from checkpoint.")
                break
            elif choice == 'D':
                delete_checkpoint(CHECKPOINT_FILE)
                delete_frame_checkpoint(FRAME_CHECKPOINT_DIR)
                break
            else:
                print("  Invalid choice. Enter C or D.")

    try:
        # Phase 0 — inventory scan (runs once per session, before any restarts)
        perception_for_scan = PerceptionModule()
        global_object_list, reference_frame = phase0_inventory_scan(pipeline, perception_for_scan)
        perception_for_scan.close()

        while True:  # outer loop: allows restart via operator [N]
            perception = PerceptionModule()

            trigger_detector = TriggerDetector(
                displacement_threshold_px=args.displacement_threshold,
                cooldown_frames=args.cooldown_frames,
                max_active_duration_ms=args.max_active_duration * 1000,
            )

            # Phase 1 — autonomous recording
            raw_buffer, frame_buffer = phase1_record(
                pipeline, perception, planner, trigger_detector,
                initial_buffer, initial_frame_buffer,
                reference_frame=reference_frame
            )
            initial_buffer = None
            initial_frame_buffer = None

            perception.close()

            if not raw_buffer:
                print("\n  No steps recorded. Restarting...")
                continue

            print(f"\n  Phase 1 complete: {len(raw_buffer)} raw observations recorded.")

            # Phase 2+3: VLM cleanup + operator validation. Loops on operator feedback.
            operator_feedback = None
            previous_result   = None
            approved          = False

            while True:
                if operator_feedback:
                    print("\n" + "=" * 55)
                    print("  PHASE 2 — VLM CLEANUP (with operator corrections)")
                    print(f"  Feedback: {operator_feedback}")
                    print("=" * 55)
                else:
                    print("\n" + "=" * 55)
                    print("  PHASE 2 — VLM CLEANUP")
                    print("  Running Gemini visual deduplication...")
                    print("=" * 55)

                cleaned_steps = planner.infer_VLM_cleanup(
                    raw_buffer, frame_buffer,
                    object_inventory=global_object_list,
                    previous_result=previous_result,
                    operator_feedback=operator_feedback,
                )

                if cleaned_steps is None:
                    print("\n  VLM cleanup failed. Retrying without feedback...")
                    operator_feedback = None
                    previous_result   = None
                    continue

                approved, feedback = phase3_validate(cleaned_steps, title="RECORDED PROCEDURE")

                if approved is None:    # [X] abort
                    print("\n  Aborted. Nothing saved.")
                    break
                elif approved:          # [Y] save
                    save_memory(cleaned_steps, args.output)
                    delete_checkpoint(CHECKPOINT_FILE)
                    delete_frame_checkpoint(FRAME_CHECKPOINT_DIR)
                    break
                elif feedback is not None:  # [R] re-run cleanup
                    operator_feedback = feedback
                    previous_result   = cleaned_steps
                    continue
                else:                   # [N] restart recording
                    break

            if approved or approved is None:
                break
            print("\n  Restarting full session...\n")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n  Session ended.")


if __name__ == "__main__":
    main()
