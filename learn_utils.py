"""
learn_utils.py — shared utilities for learn_procedure.py and learn_procedure_inverse.py.

Both learning scripts (forward and inverse) share: webcam pipeline adapter,
frame sampling, crash-recovery checkpointing, memory output, and the operator
validation dialog.  Consolidating here avoids duplication and keeps main.py's
imports clean.
"""

import cv2
import json
import os
import shutil
import numpy as np


class _WebcamColorFrame:
    def __init__(self, frame):
        self._frame = frame

    def get_data(self):
        return self._frame


class _WebcamFrameSet:
    def __init__(self, cap):
        ret, frame = cap.read()
        self._color = _WebcamColorFrame(frame) if ret else None

    def get_color_frame(self):
        return self._color


class WebcamPipeline:
    """Drop-in replacement for rs.pipeline() that reads from a webcam."""

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam at index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def wait_for_frames(self):
        return _WebcamFrameSet(self.cap)

    def stop(self):
        self.cap.release()


class _VideoColorFrame:
    def __init__(self, frame):
        self._frame = frame

    def get_data(self):
        return self._frame


class _VideoFrameSet:
    def __init__(self, color):
        self._color = color

    def get_color_frame(self):
        return self._color


class VideoPipeline:
    """Drop-in pipeline that reads from a video file.

    Adds attributes used by the learn loop to switch into video mode:
      - video_timestamp_ms(): replaces wall-clock timestamp for trigger logic
      - waitkey_delay_ms(): paces playback at video FPS (modulated by speed)
      - frame_idx, total_frames, fps: for the progress overlay
      - done: True once EOF is reached
    """

    def __init__(self, path, speed=1.0):
        if not os.path.exists(path):
            raise RuntimeError(f"Video file not found: {path}")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps          = fps if fps and fps > 0 else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.speed        = max(0.1, speed)
        self.frame_idx    = 0
        self.done         = False

    def wait_for_frames(self):
        ret, frame = self.cap.read()
        if not ret:
            self.done = True
            return _VideoFrameSet(None)
        self.frame_idx += 1
        return _VideoFrameSet(_VideoColorFrame(frame))

    def video_timestamp_ms(self):
        return (self.frame_idx / self.fps) * 1000.0

    def waitkey_delay_ms(self):
        return max(1, int(1000.0 / (self.fps * self.speed)))

    def stop(self):
        self.cap.release()


def capture_frames(pipeline, n_frames=1):
    """Capture n_frames from the pipeline immediately."""
    frames_out = []
    for _ in range(n_frames):
        fs = pipeline.wait_for_frames()
        color = fs.get_color_frame()
        if color:
            frames_out.append(np.asanyarray(color.get_data()))
    return frames_out


def extract_action_frames(active_buffer):
    """
    Sample start, mid_1, mid_2 from the active-phase buffer (at 0, 1/3, 2/3).
    Sampling at 1/3 and 2/3 gives the VLM action-in-progress frames, not just endpoints.
    """
    n = len(active_buffer)
    start_frame = active_buffer[0]

    if n >= 3:
        mid_frames = [active_buffer[n // 3], active_buffer[2 * n // 3]]
    elif n == 2:
        mid_frames = [active_buffer[0], active_buffer[1]]
    else:
        mid_frames = [active_buffer[0], active_buffer[0]]

    return start_frame, mid_frames[0], mid_frames[1]


def checkpoint_buffer(raw_buffer, path):
    """Persist raw observation list to a JSON checkpoint file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(raw_buffer, f, indent=2)


def load_checkpoint(path):
    """Return checkpoint list if valid, else None."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return None


def delete_checkpoint(path):
    if os.path.exists(path):
        os.remove(path)


def checkpoint_frames(step_index, frames, directory):
    """Write burst frames to disk keyed by step index."""
    os.makedirs(directory, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(directory, f"step_{step_index:03d}_{i}.jpg"), frame)


def load_frame_checkpoint(raw_buffer, directory):
    """Reconstruct frame_buffer list from saved JPEG files."""
    frame_buffer = []
    if not os.path.isdir(directory):
        return frame_buffer

    for obs in raw_buffer:
        step_idx = obs.get("raw_step_index", 0)
        frames, i = [], 0
        while True:
            p = os.path.join(directory, f"step_{step_idx:03d}_{i}.jpg")
            if not os.path.exists(p):
                break
            img = cv2.imread(p)
            if img is not None:
                frames.append(img)
            i += 1
        if frames:
            frame_buffer.append({"step_index": step_idx, "frames": frames})

    return frame_buffer


def delete_frame_checkpoint(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory)


def render_analyzing_overlay(annotated_frame, vlm_frames):
    """Composite the frozen annotated frame with a yellow ANALYZING banner and
    a thumbnail strip of the 4 frames being sent to the VLM."""
    h, w = annotated_frame.shape[:2]
    banner_h = 40
    thumb_h  = 110

    composite = np.zeros((h + banner_h + thumb_h, w, 3), dtype=np.uint8)
    composite[:h] = annotated_frame

    cv2.rectangle(composite, (0, h), (w, h + banner_h), (0, 200, 255), -1)
    cv2.putText(composite, "ANALYZING REMOVAL  -  VLM reasoning...",
                (10, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    n = len(vlm_frames)
    if n > 0:
        thumb_w = w // n
        labels  = ["start", "mid 1", "mid 2", "end"]
        for i, f in enumerate(vlm_frames):
            t = cv2.resize(f, (thumb_w, thumb_h))
            composite[h + banner_h:h + banner_h + thumb_h,
                      i * thumb_w:(i + 1) * thumb_w] = t
            label = labels[i] if i < len(labels) else str(i)
            cv2.putText(composite, label,
                        (i * thumb_w + 6, h + banner_h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(composite, label,
                        (i * thumb_w + 6, h + banner_h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    return composite


def draw_video_progress(frame, pipeline):
    """Draw a green progress bar + mm:ss timecode if pipeline is a VideoPipeline."""
    if not hasattr(pipeline, 'frame_idx') or not hasattr(pipeline, 'total_frames'):
        return
    h, w = frame.shape[:2]
    total = max(1, pipeline.total_frames)
    pct   = min(1.0, pipeline.frame_idx / total)
    cv2.rectangle(frame, (0, h - 4), (w, h),               (50, 50, 50), -1)
    cv2.rectangle(frame, (0, h - 4), (int(w * pct), h),    (0, 200, 0), -1)
    t_cur = pipeline.frame_idx     / pipeline.fps
    t_tot = pipeline.total_frames  / pipeline.fps
    timecode = (f"{int(t_cur//60):02d}:{int(t_cur%60):02d} / "
                f"{int(t_tot//60):02d}:{int(t_tot%60):02d}")
    cv2.putText(frame, timecode, (w - 160, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def save_memory(steps, output_path):
    """Write assembly steps to memory.json."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(steps, f, indent=4, ensure_ascii=False)
    print(f"\n  Saved {len(steps)} steps -> {output_path}")


def phase3_validate(cleaned_steps, title="RECORDED PROCEDURE"):
    """
    Display cleaned steps and prompt operator.
    Returns: (True,None) approve | (False,str) re-run with corrections | (False,None) restart | (None,None) abort.
    """
    print("\n" + "=" * 55)
    print(f"  {title} ({len(cleaned_steps)} steps)")
    print("=" * 55)

    for step in cleaned_steps:
        print(f"  {step.get('step number', '?')}. {step.get('step description', 'unknown')}")
        print(f"     Objects: {', '.join(step.get('objects_required', []))}")

    print("=" * 55)
    print("  [Y] Save and exit")
    print("  [R] Re-run cleanup with corrections (same recordings)")
    print("  [N] Restart full recording from scratch")
    print("  [X] Abort — discard everything and exit without saving")
    print("=" * 55)

    while True:
        choice = input("\n  Your choice: ").strip().upper()
        if choice == 'Y':
            return True, None
        if choice == 'R':
            print("\n  Describe what needs to be corrected (e.g. 'step 3 should")
            print("  include screws', 'merge steps 2 and 3', 'remove step 4'):")
            feedback = input("  > ").strip()
            if not feedback:
                print("  No feedback provided. Please try again.")
                continue
            return False, feedback
        if choice == 'N':
            return False, None
        if choice == 'X':
            return None, None
        print("  Invalid choice. Enter Y, R, N, or X.")
