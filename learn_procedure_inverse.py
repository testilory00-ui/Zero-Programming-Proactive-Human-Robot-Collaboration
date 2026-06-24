"""
LEARN PROCEDURE INVERSE — Learn Assembly by Observing Disassembly

Watches an operator disassemble a product, records raw observations of each
removal action via Gemini VLM, then uses Gemini to invert and reorder the
observations into a structured assembly procedure (memory.json).

There is no Phase 0 inventory scan: objects appear incrementally as the
operator removes them, and the inventory is built on the fly from YOLO
confirmed_objects.

Usage:
  python learn_procedure_inverse.py
  python learn_procedure_inverse.py --serial <camera_serial>
  python learn_procedure_inverse.py --webcam [--webcam-index 1]
  python learn_procedure_inverse.py --video path/to/disassembly.mp4 [--speed 1.0]
  python learn_procedure_inverse.py --output learned_memory.json
  python learn_procedure_inverse.py --cooldown-frames 90

Required: GEMINI_API_KEY environment variable.
"""

import cv2
import numpy as np
import argparse
import os

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
    WebcamPipeline, VideoPipeline,
    capture_frames, extract_action_frames,
    checkpoint_buffer, load_checkpoint, delete_checkpoint,
    checkpoint_frames, load_frame_checkpoint, delete_frame_checkpoint,
    save_memory, phase3_validate,
    render_analyzing_overlay, draw_video_progress,
)

CHECKPOINT_FILE      = "learn_procedure_inverse_checkpoint.json"
FRAME_CHECKPOINT_DIR = "learn_frames_inverse"


class InverseTriggerDetector:
    """
    Detects when a disassembly action has completed.

    Two conditions must both be met for a trigger to fire:
      1. Hand state downgrade: from "assembly" to "pinch" or "nothing"
         (operator finished manipulating and released the component).
      2. New object appearance: a YOLO-confirmed object class that was not
         present before the action appears in the scene — the removed piece
         is now visible as a separate object on the table.

    This two-condition design avoids false triggers from hand movements that
    don't actually remove anything from the assembly.

    State machine: IDLE → ACTIVE → CHECK_APPEARANCE → COOLDOWN → IDLE

    Fallback: trigger fires (via CHECK_APPEARANCE) if the operator stays in
    ACTIVE longer than max_active_duration_ms.
    """

    IDLE             = "idle"
    ACTIVE           = "active"
    CHECK_APPEARANCE = "check_appearance"
    COOLDOWN         = "cooldown"

    STATE_LEVEL = {"nothing": 0, "pinch": 1, "assembly": 2}

    def __init__(self, min_active_duration_ms=300,
                 cooldown_frames=90, max_active_duration_ms=20000,
                 check_appearance_grace_frames=45,
                 release_confirm_frames=5,
                 video_mode=False):
        # In video playback the perception model tends to classify most frames as
        # "assembly", so consecutive_below_assembly rarely increments.  Use a much
        # shorter max_active window and a single-frame release threshold so the
        # trigger fires within a few seconds instead of waiting 20 s.
        if video_mode:
            max_active_duration_ms = min(max_active_duration_ms, 5000)
            release_confirm_frames = 1
        self.min_active_duration_ms          = min_active_duration_ms
        self.cooldown_frames                 = cooldown_frames
        self.max_active_duration_ms          = max_active_duration_ms
        self.check_appearance_grace_frames   = check_appearance_grace_frames
        self.release_confirm_frames          = release_confirm_frames
        self.known_classes                   = set()
        self.reset()

    def reset(self):
        self.state                        = self.IDLE
        self.active_since_ms              = None
        self.pre_action_counts            = {}
        self.cooldown_remaining           = 0
        self.check_appearance_remaining   = 0
        self.consecutive_below_assembly   = 0

    @staticmethod
    def _class_counts(confirmed_objects):
        """Return {class_name: instance_count} from confirmed_objects.
        Keys in confirmed_objects are (class_name, track_id) tuples."""
        counts = {}
        for cls, _ in confirmed_objects:
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    @staticmethod
    def _confirmed_classes(confirmed_objects):
        return {cls for cls, _ in confirmed_objects}

    def _check_new_object(self, confirmed_objects):
        """
        Return (class_name, reason) if a new object appeared since pre_action_counts, else (None, None).
        Checks both new class names and increased instance counts (robust to ID reassignment).
        """
        current_counts  = self._class_counts(confirmed_objects)
        baseline_classes = set(self.pre_action_counts.keys()) | self.known_classes

        for cls in current_counts:
            if cls not in baseline_classes:
                return cls, f"new object class: {cls}"

        for cls, count in current_counts.items():
            pre_count = self.pre_action_counts.get(cls, 0)
            if count > pre_count:
                return cls, f"new instance of {cls} ({pre_count} → {count})"

        return None, None

    def _enter_check_appearance(self, confirmed_objects, reason_prefix):
        """Transition to CHECK_APPEARANCE and do an immediate new-object check."""
        print(f"  [Trigger] ACTIVE → CHECK_APPEARANCE  ({reason_prefix})")
        self.state = self.CHECK_APPEARANCE
        self.check_appearance_remaining = self.check_appearance_grace_frames
        new_cls, reason = self._check_new_object(confirmed_objects)
        if new_cls:
            print(f"  [Trigger] CHECK_APPEARANCE → COOLDOWN  ({reason})")
            self._fire(confirmed_objects)
            return True
        return False

    def _handle_active(self, hand_state, timestamp_ms, current_level, confirmed_objects):
        duration = timestamp_ms - (self.active_since_ms or timestamp_ms)

        if current_level >= 2:
            self.consecutive_below_assembly = 0
        else:
            self.consecutive_below_assembly += 1

        if duration >= self.max_active_duration_ms:
            return self._enter_check_appearance(
                confirmed_objects, f"max duration {duration/1000:.1f}s reached")

        if self.consecutive_below_assembly >= self.release_confirm_frames:
            if duration < self.min_active_duration_ms:
                self.state = self.IDLE
                return False
            return self._enter_check_appearance(
                confirmed_objects, f"assembly → {hand_state} confirmed")

        return False

    def _handle_check_appearance(self, confirmed_objects):
        new_cls, reason = self._check_new_object(confirmed_objects)
        if new_cls:
            print(f"  [Trigger] CHECK_APPEARANCE → COOLDOWN  ({reason})")
            self._fire(confirmed_objects)
            return True
        self.check_appearance_remaining -= 1
        if self.check_appearance_remaining <= 0:
            print("  [Trigger] CHECK_APPEARANCE → IDLE  "
                  "(no new object confirmed, discarding action)")
            self.state = self.IDLE
        return False

    def update(self, log_entry, confirmed_objects):
        """Process one frame. Returns True when a trigger fires."""
        hand_state    = log_entry.get("instantaneous_hand_state", log_entry["hand_state"])
        timestamp_ms  = log_entry["timestamp"]
        current_level = self.STATE_LEVEL.get(hand_state, 0)

        if self.state == self.COOLDOWN:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining <= 0:
                self.state = self.IDLE
            return False

        if self.state == self.IDLE:
            if current_level >= 2:
                self.state               = self.ACTIVE
                self.active_since_ms     = timestamp_ms
                self.consecutive_below_assembly = 0
                self.pre_action_counts   = self._class_counts(confirmed_objects)
            return False

        if self.state == self.ACTIVE:
            return self._handle_active(hand_state, timestamp_ms, current_level, confirmed_objects)

        if self.state == self.CHECK_APPEARANCE:
            return self._handle_check_appearance(confirmed_objects)

        return False

    def _fire(self, confirmed_objects):
        """Transition to COOLDOWN; update known_classes so the same class doesn't re-trigger."""
        self.known_classes.update(self._confirmed_classes(confirmed_objects))
        self.state              = self.COOLDOWN
        self.cooldown_remaining = self.cooldown_frames


MAX_DISPLAY_H = 720  # cap display height so the window fits on screen


def _fit_display(frame, max_h=MAX_DISPLAY_H):
    h, w = frame.shape[:2]
    if h <= max_h:
        return frame
    scale = max_h / h
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def phase1_record_inverse(pipeline, perception, planner, trigger_detector,
                          initial_buffer=None, initial_frame_buffer=None):
    """
    Record disassembly observations autonomously.
    Returns (raw_buffer, frame_buffer, object_inventory).
    """
    raw_buffer   = initial_buffer       if initial_buffer       else []
    frame_buffer = initial_frame_buffer if initial_frame_buffer else []
    step_count   = len(raw_buffer)

    # Inventory is built from YOLO confirmed_objects each frame (not from VLM results).
    all_objects_seen    = set()
    active_frame_buffer = []
    MAX_ACTIVE_FRAMES   = 60  # ~2 s at 30 fps

    print("\n" + "=" * 55)
    print("  PHASE 1 — DISASSEMBLY RECORDING")
    print("  Disassemble the product. Press Q when done.")
    if step_count > 0:
        print(f"  Resuming from checkpoint ({step_count} steps already recorded)")
    print("=" * 55 + "\n")

    is_video = hasattr(pipeline, 'video_timestamp_ms')

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            if getattr(pipeline, 'done', False):
                print("\n  End of video reached.")
                break
            continue

        frame = np.asanyarray(color_frame.get_data())
        log_entry, annotated_frame = perception.process_frame(frame)

        if is_video:
            log_entry["timestamp"] = pipeline.video_timestamp_ms()

        all_objects_seen.update(key[0] for key in perception.confirmed_objects)

        # Clear buffer when a new ACTIVE phase begins (IDLE → ACTIVE).
        was_idle  = trigger_detector.state == InverseTriggerDetector.IDLE
        triggered = trigger_detector.update(log_entry, perception.confirmed_objects)
        if was_idle and trigger_detector.state == InverseTriggerDetector.ACTIVE:
            active_frame_buffer.clear()

        # Accumulate frames during ACTIVE; buffer stays alive in CHECK_APPEARANCE
        # so frames are still available if the trigger fires there.
        if trigger_detector.state == InverseTriggerDetector.ACTIVE:
            active_frame_buffer.append(frame)
            if len(active_frame_buffer) > MAX_ACTIVE_FRAMES:
                active_frame_buffer.pop(0)

        if triggered:
            if not perception.confirmed_objects:
                print("  [Skip] Trigger fired but no confirmed objects in scene.")
                active_frame_buffer.clear()
            else:
                print(f"\n  [Trigger] Removal detected ({len(active_frame_buffer)} active frames).")

                start_frame, mid1, mid2 = extract_action_frames(active_frame_buffer)
                active_frame_buffer.clear()

                end_frames = capture_frames(pipeline, n_frames=1)
                end_frame  = end_frames[0] if end_frames else frame

                # 4-frame sequence: before / removing / removing / after
                learn_frames   = [start_frame, mid1, mid2, end_frame]
                cleanup_frames = [mid1, mid2]

                # YOLO reference from the post-removal scene anchors object names for the VLM.
                results = perception.object_model(end_frame, verbose=False)
                reference_frame  = None
                detected_classes = []
                if results and results[0].boxes:
                    reference_frame  = results[0].plot()
                    cls_ids          = results[0].boxes.cls.cpu().numpy()
                    detected_classes = sorted(set(
                        perception.class_names[int(c)] for c in cls_ids
                    ))

                print("\n  [VLM] Analyzing removal action...")

                overlay = render_analyzing_overlay(annotated_frame, learn_frames)
                cv2.imshow("Learn (Inverse) — Perception", _fit_display(overlay))
                cv2.waitKey(1)

                vlm_result = planner.infer_VLM_learn_inverse(
                    learn_frames,
                    reference_frame=reference_frame,
                    detected_classes=detected_classes,
                )

                if vlm_result:
                    step_count += 1
                    observation = {
                        "timestamp_ms":     log_entry["timestamp"],
                        "description":      vlm_result.get("description", ""),
                        "objects_required": vlm_result.get("objects_required", []),
                        "removed_from":     vlm_result.get("removed_from", ""),
                        "raw_step_index":   step_count,
                    }
                    raw_buffer.append(observation)
                    frame_buffer.append({"step_index": step_count, "frames": cleanup_frames})

                    checkpoint_buffer(raw_buffer, CHECKPOINT_FILE)
                    checkpoint_frames(step_count, cleanup_frames, FRAME_CHECKPOINT_DIR)

                    print(f"  [Step {step_count}] {observation['description']}")
                    print(f"              Objects: {', '.join(observation['objects_required'])}")
                    print(f"              Removed from: {observation['removed_from']}\n")

                    if is_video:
                        flash = end_frame.copy()
                        cv2.rectangle(flash, (0, 0), (flash.shape[1], 60), (0, 180, 0), -1)
                        desc = observation['description'][:70]
                        cv2.putText(flash, f"Step {step_count}: {desc}",
                                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (255, 255, 255), 2)
                        cv2.imshow("Learn (Inverse) — Perception", _fit_display(flash))
                        cv2.waitKey(800)
                else:
                    print("  [VLM] No result returned. Skipping.")

        trigger_colors = {
            "idle": (180, 180, 180), "active": (0, 255, 0),
            "check_appearance": (0, 255, 255), "cooldown": (0, 165, 255),
        }
        t_color = trigger_colors.get(trigger_detector.state, (200, 200, 200))
        cv2.putText(annotated_frame, f"Steps: {step_count}",
                    (10, annotated_frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Trigger: {trigger_detector.state.upper()}",
                    (10, annotated_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, t_color, 1)
        hint = "Q stop  |  SPACE pause" if is_video else "Q to stop"
        cv2.putText(annotated_frame, hint,
                    (annotated_frame.shape[1] - 220, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if is_video:
            draw_video_progress(annotated_frame, pipeline)

        cv2.imshow("Learn (Inverse) — Perception", _fit_display(annotated_frame))

        delay = pipeline.waitkey_delay_ms() if is_video else 1
        key = cv2.waitKey(delay) & 0xFF

        if is_video and key == ord(' '):
            while True:
                k2 = cv2.waitKey(50) & 0xFF
                if k2 != 255:
                    key = k2
                    break

        quit_requested = key in (ord('q'), ord('Q'))

        if not quit_requested and _HAS_MSVCRT:
            while msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch.lower() == b'q':
                    quit_requested = True

        if quit_requested:
            break

    cv2.destroyAllWindows()
    object_inventory = sorted(all_objects_seen)
    return raw_buffer, frame_buffer, object_inventory


def run_cleanup_loop(planner, raw_buffer, frame_buffer, object_inventory, output_path):
    """Run VLM inverse-cleanup and operator validation. Returns True (saved), False ([N] restart), None ([X] abort)."""
    operator_feedback = None
    previous_result   = None

    while True:
        if operator_feedback:
            print("\n" + "=" * 55)
            print("  PHASE 2 — ASSEMBLY RECONSTRUCTION (with operator corrections)")
            print(f"  Feedback: {operator_feedback}")
            print("=" * 55)
        else:
            print("\n" + "=" * 55)
            print("  PHASE 2 — ASSEMBLY RECONSTRUCTION")
            print("  Running Gemini: inverting disassembly → assembly order...")
            print("=" * 55)

        cleaned_steps = planner.infer_VLM_cleanup_inverse(
            raw_buffer, frame_buffer,
            object_inventory=object_inventory,
            previous_result=previous_result,
            operator_feedback=operator_feedback,
        )

        if cleaned_steps is None:
            print("\n  VLM cleanup failed. Retrying without feedback...")
            operator_feedback = None
            previous_result   = None
            continue

        approved, feedback = phase3_validate(
            cleaned_steps, title="RECONSTRUCTED ASSEMBLY PROCEDURE")

        if approved is None:    # [X]
            print("\n  Aborted. Nothing saved.")
            return None
        if approved:            # [Y]
            save_memory(cleaned_steps, output_path)
            delete_checkpoint(CHECKPOINT_FILE)
            delete_frame_checkpoint(FRAME_CHECKPOINT_DIR)
            return True
        if feedback is not None:    # [R]
            operator_feedback = feedback
            previous_result   = cleaned_steps
            continue
        return False            # [N]


def main():
    parser = argparse.ArgumentParser(
        description="Learn assembly procedure by observing disassembly"
    )
    parser.add_argument("--serial", type=str, default=None,
                        help="RealSense camera serial number (uses first available if omitted)")
    parser.add_argument("--webcam", action="store_true",
                        help="Use a webcam instead of a RealSense camera")
    parser.add_argument("--webcam-index", type=int, default=0,
                        help="Webcam device index (default: 0); only used with --webcam")
    parser.add_argument("--video", nargs="?", const="video_learning",
                        default="video_learning",
                        metavar="FOLDER_OR_FILE",
                        help="Folder to pick a video from (default: video_learning/) "
                             "or direct path to a video file; "
                             "ignored when --webcam or --serial is given")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier when using --video (default: 1.0)")
    parser.add_argument("--output", type=str, default="learned_memory.json",
                        help="Output file path (default: learned_memory.json)")
    parser.add_argument("--cooldown-frames", type=int, default=90,
                        help="Frames to ignore after a trigger (default: 90)")
    parser.add_argument("--max-active-duration", type=int, default=20,
                        help="Seconds before a trigger fires unconditionally "
                             "while hand is still active (default: 20)")
    args = parser.parse_args()

    if args.webcam and args.serial:
        raise SystemExit("Choose only one camera source: --webcam or --serial.")

    VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}

    def _list_videos(folder):
        if not os.path.isdir(folder):
            return []
        return sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS
        )

    def _select_video(folder):
        """List videos in folder and let the operator pick one. Returns full path."""
        videos = _list_videos(folder)
        if not videos:
            raise SystemExit(f"No videos found in '{folder}' "
                             f"(supported: {', '.join(sorted(VIDEO_EXTS))})")
        if len(videos) == 1:
            path = os.path.join(folder, videos[0])
            print(f"\nAuto-selected video: {path}")
            return path
        print(f"\nVideos found in '{folder}':")
        for i, v in enumerate(videos, 1):
            print(f"  [{i}] {v}")
        while True:
            choice = input("  Select video number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(videos):
                return os.path.join(folder, videos[int(choice) - 1])
            print(f"  Invalid choice. Enter a number between 1 and {len(videos)}.")

    if args.webcam:
        print(f"\nStarting webcam (index {args.webcam_index})...")
        pipeline = WebcamPipeline(index=args.webcam_index)
    elif args.serial:
        if not _RS_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not installed.")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(args.serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        print("\nStarting RealSense camera...")
        pipeline.start(config)
    else:
        # --video with a folder path (or default video_learning/)
        video_path = args.video
        if os.path.isdir(video_path):
            video_path = _select_video(video_path)
        elif not os.path.exists(video_path):
            raise SystemExit(f"Video not found: '{video_path}'")
        print(f"\nLoading video: {video_path} (speed x{args.speed})...")
        pipeline = VideoPipeline(video_path, speed=args.speed)
        print(f"  {pipeline.total_frames} frames @ {pipeline.fps:.1f} fps "
              f"(~{pipeline.total_frames/pipeline.fps:.1f}s)")

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
        while True:  # outer loop: allows restart via operator [N]
            perception = PerceptionModule()
            if isinstance(pipeline, VideoPipeline):
                # Video tends to keep wrists within the default 0.7 threshold
                # almost permanently → classify as "assembly" the whole time.
                # A tighter threshold forces genuine two-hand contact to trigger it.
                perception.ASSEMBLY_WRIST_THRESHOLD = 0.6

            trigger_detector = InverseTriggerDetector(
                cooldown_frames=args.cooldown_frames,
                max_active_duration_ms=args.max_active_duration * 1000,
                video_mode=isinstance(pipeline, VideoPipeline),
            )

            raw_buffer, frame_buffer, object_inventory = phase1_record_inverse(
                pipeline, perception, planner, trigger_detector,
                initial_buffer, initial_frame_buffer,
            )
            initial_buffer = None
            initial_frame_buffer = None

            perception.close()

            if not raw_buffer:
                print("\n  No steps recorded. Restarting...")
                perception = PerceptionModule()
                continue

            print(f"\n  Phase 1 complete: {len(raw_buffer)} raw observations recorded.")
            print(f"  Objects seen: {', '.join(object_inventory)}")

            result = run_cleanup_loop(planner, raw_buffer, frame_buffer,
                                      object_inventory, args.output)
            if result is True or result is None:
                break
            # result is False → operator chose [N], restart
            print("\n  Restarting full session...\n")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n  Session ended.")


if __name__ == "__main__":
    main()
