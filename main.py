"""
ASSEMBLY TASK PLANNER AND ROBOT CONTROLLE

Dual Intel RealSense cameras + LLM (Qwen/Llama) + VLM (Gemini) + ABB GoFa robot.

─── MODES ────────────────────────────────────────────────────────────────────
  python main.py                   Assembly mode (default). Requires 2 cameras.
  python main.py --disassembly     Robot clears the shared area back to remote.
                                   Only the robot camera is used; no LLM inference.

─── INFERENCE ────────────────────────────────────────────────────────────────
  --with-context                   Inject StepTracker softmax probabilities into
                                   every LLM and VLM prompt to narrow predictions.
                                   Without this flag the models receive no context.

  --local-model                    Use the local OpenVINO-quantised model (Qwen)
                                   for LLM inference instead of the Hugging Face
                                   Inference API (Llama-4 Scout, default).
                                   Local model is slower to load but runs offline.

  --with-filter                    Enable the scene-consistency filter. Every
                                   non-trivial LLM/VLM prediction is validated
                                   against YOLO-confirmed objects before dispatch.
                                   Invalid predictions are silently dropped and an
                                   augmented hint is injected into the next natural
                                   inference trigger.

─── SETUP ────────────────────────────────────────────────────────────────────
  --force-recalibrate              Recompute the workspace homography from ArUco
                                   markers even if homography.json already exists.
  --silent-setup                   Skip the GP confirmation overlay window.
                                   Loads existing homography.json automatically.
  --perception-serial <SN>         RealSense serial for the perception camera.
  --robot-serial <SN>              RealSense serial for the robot camera.
                                   (Both default to hardcoded SNs when omitted.)

─── TRIAL LOGGING ────────────────────────────────────────────────────────────
  --participant-id <ID>            Enable trial logging for participant <ID>
                                   (e.g. P001). Omit to run without any logging.
  --trial-number <N>               Trial index for this participant (default: 1).
  --log-dir <path>                 Directory for log files (default: logs/).

  Logging writes three artefacts per run:
    logs/trial_<ID>_T<N>_<condition>_summary.json      — ATCT + latency stats
    logs/trial_<ID>_T<N>_<condition>_predictions.jsonl — one record per LLM/VLM call
    logs/trial_<ID>_T<N>_<condition>_frames/           — composite JPEGs for VLM calls

  Timer control (assembly window):
    Trial timer starts automatically when setup closes (GP window confirmed).
    Press ESC to stop the timer and exit.

─── EXAMPLES ─────────────────────────────────────────────────────────────────
  python main.py --with-context --participant-id P003 --trial-number 2
  python main.py --local-model --with-context
  python main.py --disassembly --silent-setup
"""

import cv2
import json
import argparse
from collections import deque, Counter
import threading
import queue
import numpy as np
import time
import pyrealsense2 as rs

from perception import PerceptionModule
from llm import LLM_planner
from grasp_context import GraspContext
from robot_socket_client import RobotSocketClient
from step_tracker import StepTracker

try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False


def _snapshot_step_probs(tracker) -> dict:
    """Snapshot current step probabilities from a StepTracker instance.
    Returns {step_N: float} for logging. Returns {} when tracker is None."""
    if tracker is None:
        return {}
    return {f"step_{k}": v for k, v in tracker.get_step_probabilities().items()}

def normalize_class(name: str) -> str:
    """Normalise LLM object name variants to canonical YOLO class names."""
    name = name.strip().lower()
    if name in ("screw", "screws"):
        return "screw"
    return name


# Scene-consistency filter constants
_FILTER_SKIP_OPS  = {"none", "final_cleanup"}
_FILTER_SKIP_OBJS = {"screw", "screws", "screwdriver", "carburetor_body"}


def _scene_filter(plan: dict, confirmed_classes: set) -> tuple:
    """Return (is_valid, missing_objects_list).

    A plan is valid if every non-screw/screwdriver object in objects_required
    is confirmed present in the perception-camera scene (3-frame threshold).
    Plans with next_operation 'none' or 'final_cleanup' always pass.
    """
    next_op = (plan.get("next operation") or plan.get("next_operation") or "").strip().lower()
    if next_op in _FILTER_SKIP_OPS or not next_op:
        return True, []
    required = [
        normalize_class(o)
        for o in (plan.get("objects_required") or plan.get("objects required") or [])
        if normalize_class(o) not in _FILTER_SKIP_OBJS
    ]
    if not required:
        return True, []
    if not confirmed_classes:        # no scene evidence yet → conservative pass
        return True, []
    missing = [o for o in required if o not in confirmed_classes]
    return (len(missing) == 0), missing


# Per-object Z offsets (mm above nominal GP height). 0 = flat/default.
OBJECT_Z_OFFSETS: dict[str, int] = {
    "float_bowl":       20,
    "diaphragm":        -3,
    "cover":            -1,
    "carburetor_body":  70,
    "throttle_stop":     2,
}

llm_task_queue      = queue.Queue(maxsize=1)
llm_result_queue    = queue.Queue(maxsize=1)
robot_command_queue = queue.Queue(maxsize=1)

# Set while a task is queued OR being inferred; prevents a second submission
# while the worker has dequeued but not yet finished the current task.
llm_inference_busy = threading.Event()

shared_state = {
    "robot_is_ready": True,
    "llm_call_count": 0,
    "vlm_call_count": 0,
    "llm_times": [],   # elapsed seconds per LLM call
    "vlm_times": [],   # elapsed seconds per VLM call
}
state_lock = threading.Lock()


def llm_inference_worker(planner, use_local_model=False, trial_logger=None):
    """Blocking inference loop — runs LLM or VLM on each task from llm_task_queue."""
    while True:
        try:
            task      = llm_task_queue.get()
            task_type        = task.get("type")
            data             = task.get("data")
            context          = task.get("step_completion_context", "")
            augmented_hint   = task.get("augmented_hint")
            frame_detections = task.get("frame_detections")
            was_hint         = augmented_hint is not None
            context_available = bool(context)

            if task_type == "vlm":
                print("\n[VLM TASK RECEIVED] Processing frames...")
                if was_hint:
                    print(f"[VLM] Hint-injected call: {augmented_hint[:80]}...")
                t0      = time.time()
                result  = planner.infer_VLM(data, step_completion_context=context,
                                            model='gemini-3.1-flash-lite',
                                            augmented_hint=augmented_hint,
                                            frame_detections=frame_detections)
                # gemini-3.1-flash-lite-preview
                elapsed = time.time() - t0
                with state_lock:
                    shared_state["vlm_call_count"] += 1
                    shared_state["vlm_times"].append(elapsed)
                print(f"[VLM] Done in {elapsed:.1f}s")

                if trial_logger is not None and result is not None:
                    try:
                        vlm_output = json.loads(result) if isinstance(result, str) else result
                        step_probs = task.get("step_probs", {})
                        trial_logger.log_vlm_call(
                            frames             = data,
                            step_probabilities = step_probs,
                            vlm_output         = vlm_output,
                            inference_time     = elapsed,
                            context_available  = context_available,
                            was_hint_injection = was_hint,
                        )
                    except Exception as log_err:
                        print(f"[TrialLogger] VLM log error (non-fatal): {log_err}")

            else:
                print("\n[LLM TASK RECEIVED] Processing scene...")
                if was_hint:
                    print(f"[LLM] Hint-injected call: {augmented_hint[:80]}...")
                t0      = time.time()
                scene   = {**data, "step_completion_context": context}
                if use_local_model:
                    result = planner.infer_LLM(scene, augmented_hint=augmented_hint)
                else:
                    result = planner.infer_LLM_HF(scene, augmented_hint=augmented_hint)
                elapsed = time.time() - t0
                with state_lock:
                    shared_state["llm_call_count"] += 1
                    shared_state["llm_times"].append(elapsed)
                print(f"[LLM] Done in {elapsed:.1f}s")

                if trial_logger is not None and result is not None:
                    try:
                        llm_output = json.loads(result) if isinstance(result, str) else result
                        step_probs = task.get("step_probs", {})
                        trial_logger.log_llm_call(
                            semantic_action    = data.get("semantic_action", ""),
                            step_probabilities = step_probs,
                            llm_output         = llm_output,
                            inference_time     = elapsed,
                            context_available  = context_available,
                            was_hint_injection = was_hint,
                        )
                    except Exception as log_err:
                        print(f"[TrialLogger] LLM log error (non-fatal): {log_err}")

            if result is None:
                print("[LLM Worker] Planner returned None — skipping result.")
                llm_inference_busy.clear()
                llm_task_queue.task_done()
                continue

            if not llm_result_queue.empty():
                try:
                    llm_result_queue.get_nowait()
                except queue.Empty:
                    pass
            llm_result_queue.put(result)
            llm_inference_busy.clear()
            llm_task_queue.task_done()

        except Exception as e:
            print(f"An error occurred in the llm_inference_worker: {e}")
            llm_inference_busy.clear()


def robot_command_worker(perception_module, lock, grasp_ctx, robot_client):
    """Resolve GP assignments from a confirmed plan and dispatch pick-and-place batches to the robot."""
    screws_in_shared   = False

    while True:
        try:
            task         = robot_command_queue.get()
            plan         = task.get("plan")
            robot_frame  = task.get("robot_frame")   # single frame (disassembly loop)
            robot_frames = task.get("robot_frames")  # list of frames (assembly loop)

            next_op = str(plan.get("next operation", plan.get("next_operation", ""))).strip().lower()

            if next_op == "none":
                print("[Robot Worker] Next operation is 'none'. No action required.")
                continue

            if next_op == "final_cleanup":
                print("[Robot Worker] Final cleanup — returning containers to remote zone.")
                cleanup_removes = []
                src = grasp_ctx.find_screws_container("shared")
                dst = grasp_ctx.find_screws_container("remote")
                if src and dst:
                    cleanup_removes.append({
                        "object_class": "screw",
                        "pick_gp_id":  src["id"],
                        "place_gp_id": dst["id"],
                        "pick_px":     src["px_grasp"],
                        "place_px":    dst["px_grasp"],
                    })
                screws_in_shared  = False
                if cleanup_removes:
                    success = robot_client.execute_batch_from_commands([], cleanup_removes)
                    if not success:
                        print("[Robot Worker] Cleanup batch failed — check FlexPendant.")
                    else:
                        print("[Robot Worker] Containers returned to remote. Robot at HomePose.")
                else:
                    print("[Robot Worker] No container GPs found — nothing to move.")
                continue

            raw_required = plan.get("objects required", plan.get("objects_required", []))

            objects_required = set(normalize_class(obj) for obj in raw_required)

            print(f"\n[Robot Worker] Plan: '{plan.get('next operation', plan.get('next_operation', ''))}' — detecting objects... ({len(robot_frames) if robot_frames else 0} frames)")
            if robot_frames:
                # min_frame_count=2: small objects (e.g. spring) may only appear in
                # 1-2 frames; requiring 3 silently drops them from the results.
                detected_objects = perception_module.detect_objects_in_frames(robot_frames, min_frame_count=2)
            else:
                # Proactive dispatch: buffer was just cleared on robot_just_finished;
                # fall back to the single fresh HomePose frame included at dispatch time.
                detected_objects = perception_module.detect_objects_in_frame(robot_frame)

            print(f"[Robot Worker] Required: {objects_required}")

            # STEP 1: assign YOLO detections to zones via nearest-GP matching.
            # Screws skip this step — their container are routed by
            # GP type in Step 3. Screwdriver uses type_filter="screwdriver" to hit
            # its dedicated GPs instead of standard slots.

            remote_area = []
            shared_area = []
            for obj in detected_objects:
                if obj['class'] == "screw":
                    continue
                bbox_center = (
                    (obj['coords'][0] + obj['coords'][2]) / 2,
                    (obj['coords'][1] + obj['coords'][3]) / 2
                )

                gp_type_filter = "screwdriver" if obj['class'] == "screwdriver" else "standard"
                gp_shared  = grasp_ctx.find_nearest_gp(bbox_center, zone="shared", type_filter=gp_type_filter)
                gp_remote  = grasp_ctx.find_nearest_gp(bbox_center, zone="remote", type_filter=gp_type_filter)

                if gp_remote:
                    remote_area.append({"obj": obj, "source_gp": gp_remote})
                    print(f"[Robot Worker] {obj['class']} → remote GP {gp_remote['id']}")
                elif gp_shared:
                    shared_area.append({"obj": obj, "source_gp": gp_shared})
                    print(f"[Robot Worker] {obj['class']} → shared GP {gp_shared['id']}")
                else:
                    print(f"[Robot Worker] {obj['class']} not near any GP")

            # STEP 2: separate objects into bring (remote→shared) and remove (shared→remote)

            objects_to_bring = [item for item in remote_area
                                 if item['obj']['class'] in objects_required
                                 and item['obj']['class'] != "screw"]
            objects_to_remove = [item for item in shared_area
                                  if item['obj']['class'] not in objects_required
                                  and item['obj']['class'] != "screw"]

            # STEP 3: compute pick-and-place GP pairs.
            bring_commands = []
            remove_commands = []
            occupied_gp_ids = set()
            occupied_gp_ids.update(item["source_gp"]["id"] for item in remote_area)
            # Only block shared GPs for items being removed (not in objects_required).
            # Items in objects_required that are already in shared stay in place and must
            # not block bring targets — e.g. carburetor_body near CTR_GENERAL_S2 would
            # otherwise prevent the spring container from being routed there.
            occupied_gp_ids.update(
                item["source_gp"]["id"] for item in shared_area
                if item["obj"]["class"] not in objects_required
            )

            # TRACK A: standard objects (not screwdriver) — routed by YOLO bbox.
            # BRING pass 1 (remote → shared): standard objects only, screwdriver deferred.
            screwdriver_bring_item = None
            for item in objects_to_bring:
                source_gp = item["source_gp"]
                obj_class = item["obj"]["class"]

                if obj_class == "screwdriver":
                    screwdriver_bring_item = item  # handled after standard objects
                    continue

                if source_gp["type"] == "general_container":
                    target_gp = grasp_ctx.find_paired_general_container(source_gp["id"])
                    if not target_gp:
                        print(f"[Robot Worker] SKIP bring {obj_class}: no paired container for {source_gp['id']}")
                        continue
                    if target_gp["id"] in occupied_gp_ids:
                        print(f"[Robot Worker] SKIP bring {obj_class}: paired container {target_gp['id']} occupied")
                        continue
                else:
                    free_shared = grasp_ctx.get_free_gps("shared", occupied_gp_ids, type_filter="standard")
                    if not free_shared:
                        print(f"[Robot Worker] SKIP bring {obj_class}: no free shared GP")
                        continue
                    target_gp = min(free_shared.values(),
                        key=lambda gp: (
                            (gp["px_center"][0] - source_gp["px_center"][0])**2 +
                            (gp["px_center"][1] - source_gp["px_center"][1])**2
                        ))
                bring_commands.append({
                    "object_class": obj_class,
                    "pick_gp_id":  source_gp["id"],
                    "place_gp_id": target_gp["label"],
                    "pick_px":     source_gp["px_grasp"],
                    "place_px":    target_gp["px_grasp"],
                    "z_offset":    OBJECT_Z_OFFSETS.get(obj_class, 0),
                })
                occupied_gp_ids.add(target_gp["label"])
                occupied_gp_ids.discard(source_gp["id"])

            # BRING pass 2: screwdriver — after standard objects, before screws.
            if screwdriver_bring_item is not None:
                source_gp = screwdriver_bring_item["source_gp"]
                target_gp = grasp_ctx.find_screwdriver_gp("shared")
                if target_gp:
                    bring_commands.append({
                        "object_class": "screwdriver",
                        "pick_gp_id":  source_gp["id"],
                        "place_gp_id": target_gp["label"],
                        "pick_px":     source_gp["px_grasp"],
                        "place_px":    target_gp["px_grasp"],
                        "z_offset":    OBJECT_Z_OFFSETS.get("screwdriver", 0),
                    })
                    occupied_gp_ids.add(target_gp["label"])
                    occupied_gp_ids.discard(source_gp["id"])
                else:
                    print("[Robot Worker] SKIP bring screwdriver: no shared screwdriver GP found")

            # REMOVE (shared → remote): mirror of above with remote as target.
            for item in objects_to_remove:
                source_gp = item["source_gp"]
                obj_class = item["obj"]["class"]

                if source_gp["type"] == "general_container":
                    target_gp = grasp_ctx.find_paired_general_container(source_gp["id"])
                    if not target_gp or target_gp["id"] in occupied_gp_ids:
                        continue
                elif obj_class == "screwdriver":
                    # Force screwdriver to its dedicated GP in the remote area
                    target_gp = grasp_ctx.find_screwdriver_gp("remote")
                else:
                    free_remote = grasp_ctx.get_free_gps("remote", occupied_gp_ids, type_filter="standard")
                    if not free_remote: continue
                    target_gp = min(free_remote.values(),
                        key=lambda gp: (
                            (gp["px_center"][0] - source_gp["px_center"][0])**2 +
                            (gp["px_center"][1] - source_gp["px_center"][1])**2
                        ))

                if not target_gp: continue

                remove_commands.append({
                    "object_class": obj_class,
                    "pick_gp_id":  source_gp["id"],
                    "place_gp_id": target_gp["label"],
                    "pick_px":     source_gp["px_grasp"],
                    "place_px":    target_gp["px_grasp"],
                    "z_offset":    OBJECT_Z_OFFSETS.get(obj_class, 0),
                })
                occupied_gp_ids.add(target_gp["label"])
                occupied_gp_ids.discard(source_gp["id"])

            # TRACK B: screws container — routed by GP type, not YOLO bbox.
            # Snapshot flags before the batch so failures can restore pre-cycle state.
            pre_batch_screws   = screws_in_shared

            if "screw" in objects_required:
                if not screws_in_shared:
                    src = grasp_ctx.find_screws_container("remote")  # CTR_SCREW_R
                    dst = grasp_ctx.find_screws_container("shared")  # CTR_SCREW_S
                    if src and dst:
                        bring_commands.append({
                            "object_class": "screw",
                            "pick_gp_id":  src["id"],
                            "place_gp_id": dst["id"],
                            "pick_px":     src["px_grasp"],
                            "place_px":    dst["px_grasp"],
                        })
                        occupied_gp_ids.add(dst["id"])
                        occupied_gp_ids.discard(src["id"])
                        screws_in_shared = True

            else:  # screw not required — return container if it was brought earlier
                if screws_in_shared:
                    src = grasp_ctx.find_screws_container("shared")  # CTR_SCREW_S
                    dst = grasp_ctx.find_screws_container("remote")  # CTR_SCREW_R
                    if src and dst:
                        remove_commands.append({
                            "object_class": "screw",
                            "pick_gp_id":  src["id"],
                            "place_gp_id": dst["id"],
                            "pick_px":     src["px_grasp"],
                            "place_px":    dst["px_grasp"],
                        })
                        screws_in_shared = False

            # STEP 4: send batch to ABB.
            success = robot_client.execute_batch_from_commands(bring_commands, remove_commands)
            if not success:
                print("[Robot Worker] Batch failed or timed out — check FlexPendant.")
                screws_in_shared  = pre_batch_screws   # restore; assume nothing moved
            else:
                print("[Robot Worker] Batch done — robot at HomePose.")

        except Exception as e:
            print(f"An error occurred in the robot_command_worker: {e}")

        finally:
            with lock:
                shared_state["robot_is_ready"] = True
            print("[Robot Worker] At HOME_OBS — ready for next task.")
            robot_command_queue.task_done()


def run_assembly_loop(
        pipeline_perception, pipeline_robot, perception,
        robot_client, grasp_ctx, state_lock, shared_state,
        step_tracker=None,
        trial_logger=None,
        filter_enabled=False,
        assembly_memory=None):
    """Assembly main loop. Pass a StepTracker to inject step-completion context
    into LLM/VLM calls; None runs without context.
    Pass filter_enabled=True (--with-filter) to validate predictions against the
    YOLO-confirmed scene before dispatch."""
    data_buffer        = deque(maxlen=90) # Buffer for assembly state - perception camera frames
    robot_frame_buffer = deque(maxlen=20) # Buffer for objects tracking - robot camera frames

    CONFIRMATION_THRESHOLD    = 1  # Threshold for confirming action candidate (single prediction is sufficient)
    STARTUP_CONFIRMATION      = 2  # Higher threshold before first dispatch: prevents a single noisy VLM result from locking in the wrong step
    MIN_FRAMES_FOR_INFERENCE  = 20 # Fast trigger when robot just finished (observe ~0.7s)
    MIN_FRAMES_DURING_ROBOT   = 180 # Slower trigger while robot is executing (observe ~6s to skip grasping phase)
    VLM_MIN_INTERVAL_DURING_ROBOT = 5.0  # Min seconds between VLM calls while robot is busy (throttles bursts)
    ROBOT_BUFFER_MIN_DISPATCH = 10  # min fresh HomePose frames before dispatching (~167ms); LLM still runs in parallel

    next_task_candidate          = None
    candidate_confirmation_count = 0
    confirmed_task_plan          = None
    first_plan_dispatched        = False   # startup: use STARTUP_CONFIRMATION until first dispatch
    prev_robot_ready             = True
    last_submitted_action        = None   # dedup: last semantic_action sent to LLM
    last_llm_result              = None   # dedup: cached LLM response for replay
    last_recorded_result         = None   # dedup: last result fed to step_tracker
    last_vlm_fire_time           = 0.0    # throttle: timestamp of last VLM dispatch to worker

    last_robot_detections: list = []
    home_pose_capture_done = False   # ensure one snapshot before first robot move
    inference_suppressed   = False   # tracks whether last-step gate has already been logged
    workspace_clear        = False   # set when robot returns home and YOLO finds no objects
    skip_first_visibility  = True    # the first robot return happens when most signature
                                     # objects are still in the remote zone (out of camera
                                     # FOV); their absence would be misread as "consumed"
                                     # and falsely flip step 1 to "done"

    # Scene-consistency filter state
    pending_augmented_hint  = None   # built on validation failure; injected into next enqueue
    last_enqueued_type      = None   # "llm" or "vlm" — type of most recently submitted task
    last_enqueued_context   = ""     # step_completion_context of most recently submitted task
    last_enqueued_has_hint  = False  # True if most recent task carried an augmented_hint
    last_confirmed_call_id  = None   # call_id (from trial_logger) of last confirmation

    ROBOT_DISP_W, ROBOT_DISP_H = 640, 360
    cv2.destroyAllWindows()
    cv2.waitKey(1)   # flush pending destroy events on Windows before creating new windows
    cv2.namedWindow("Assembly — Robot Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Assembly — Perception",   cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Assembly — Robot Camera", ROBOT_DISP_W, ROBOT_DISP_H)
    cv2.resizeWindow("Assembly — Perception",   640, 480)
    cv2.moveWindow("Assembly — Robot Camera",   0,                0)
    cv2.moveWindow("Assembly — Perception",     ROBOT_DISP_W + 10, 0)

    while True:
        frames_p      = pipeline_perception.wait_for_frames()
        color_frame_p = frames_p.get_color_frame()

        frames_r      = pipeline_robot.wait_for_frames()
        color_frame_r = frames_r.get_color_frame()

        if not color_frame_p:
            continue
        if not color_frame_r:
            print("[Main] WARNING: Robot camera frame not received — skipping.")
            continue

        frame_perception = np.asanyarray(color_frame_p.get_data())
        frame_robot      = np.asanyarray(color_frame_r.get_data())
        robot_frame_buffer.append(frame_robot.copy())

        robot_is_ready = False
        with state_lock:
            robot_is_ready = shared_state["robot_is_ready"]

        robot_just_finished = robot_is_ready and not prev_robot_ready
        prev_robot_ready    = robot_is_ready

        # Flush buffer when robot returns so detect_objects_in_frames only sees
        # clean HomePose frames — not frames captured while the arm was moving.
        if robot_just_finished:
            robot_frame_buffer.clear()

        # YOLO on robot frame only at HomePose (startup + every return) to avoid
        # captures where the arm occludes the workspace.
        if robot_just_finished or not home_pose_capture_done:
            last_robot_detections = perception.detect_objects_in_frame(frame_robot)
            home_pose_capture_done = True
            # Skip visibility evidence at startup: cold-camera misses could signal
            # false "step done" before any assembly has occurred. Also skip the
            # first robot return: at that moment objects for steps 2..N are still
            # in the remote zone (not in the robot camera's FOV), so their
            # apparent absence is structural, not evidence of completion.
            if step_tracker is not None and robot_just_finished:
                if skip_first_visibility:
                    skip_first_visibility = False
                else:
                    visible_classes = {d["class"] for d in last_robot_detections}
                    step_tracker.record_object_visibility(visible_classes)
            # Workspace-empty gate: suppress inference when all objects are gone.
            if robot_just_finished:
                if len(last_robot_detections) == 0:
                    if not workspace_clear:
                        print("[Main] Workspace empty after robot home — assembly complete, suppressing inference.")
                    workspace_clear     = True
                    confirmed_task_plan = None
                else:
                    workspace_clear = False

        frame_data, annotated_perception = perception.process_frame(frame_perception)
        data_buffer.append((frame_perception.copy(), frame_data))

        try:
            llm_result_str = llm_result_queue.get_nowait()
            last_llm_result = llm_result_str
            llm_plan        = json.loads(llm_result_str)
            suggested_next_op = llm_plan.get("next operation",
                                              llm_plan.get("next_operation"))

            # Scene-consistency filter gate — runs before confirmation counter.
            if filter_enabled:
                if robot_frame_buffer:
                    _robot_dets = perception.detect_objects_in_frames(
                        list(robot_frame_buffer), min_frame_count=1)
                    confirmed_classes = {d["class"] for d in _robot_dets}
                else:
                    confirmed_classes = set()
                is_valid, missing = _scene_filter(llm_plan, confirmed_classes)
                if not is_valid:
                    _next_op_str = suggested_next_op or "?"
                    _missing_str = '", "'.join(missing[:3])
                    _present_str = (", ".join(sorted(confirmed_classes))
                                    if confirmed_classes else "none detected")
                    # Compute which steps are compatible with the objects actually on the table.
                    # Separate step-number labels (for logging) from exact descriptions (for the hint,
                    # so the model can copy the value directly into the JSON without reformatting).
                    _compatible_labeled = []
                    _compatible_descs   = []
                    if assembly_memory is not None:
                        for _s in assembly_memory:
                            _req = [normalize_class(o) for o in _s.get("objects_required", [])
                                    if normalize_class(o) not in _FILTER_SKIP_OBJS]
                            if not _req or all(o in confirmed_classes for o in _req):
                                _compatible_labeled.append(
                                    f"Step {_s['step number']}: {_s['step description']}"
                                )
                                _compatible_descs.append(_s["step description"])
                    _compatible_str  = "; ".join(_compatible_labeled) if _compatible_labeled else "none clearly matching"
                    _valid_next_ops  = " | ".join(f'"{d}"' for d in _compatible_descs) if _compatible_descs else '"none"'
                    _progress_line   = (f"\nAssembly progress: {step_tracker.completion_summary_for_prompt()}"
                                        if step_tracker is not None else "")
                    _hint_text = (
                        f'REJECTED: "next operation"="{_next_op_str}" is invalid — missing objects: {_missing_str}.\n'
                        f'Objects on table: {_present_str}.{_progress_line}\n'
                        f'"next operation" MUST be one of (copy exact value, earliest first): {_valid_next_ops}.\n'
                        f'"stage of assembly" = the step immediately before your chosen "next operation".'
                    )
                    print(f"[Filter] INVALID plan '{_next_op_str}' — missing: {missing}. "
                          f"Hint queued for next trigger.")
                    if trial_logger is not None:
                        trial_logger.log_validation_failure(
                            call_type          = last_enqueued_type or "llm",
                            predicted_plan     = llm_plan,
                            missing_objects    = missing,
                            confirmed_classes  = confirmed_classes,
                            context_str        = last_enqueued_context,
                            was_hint_injection = last_enqueued_has_hint,
                            augmented_hint_built = _hint_text,
                        )
                    if last_enqueued_has_hint and trial_logger is not None:
                        trial_logger.log_hint_outcome(last_enqueued_type or "llm",
                                                      "invalid_dropped")
                    pending_augmented_hint = _hint_text
                    last_submitted_action  = None
                    last_llm_result        = None
                    last_recorded_result   = None
                    last_enqueued_has_hint = False
                    data_buffer.clear()  # discard stale frames; retry with fresh perception
                    continue  # skip confirmation counter — invalid plan does not confirm
                else:
                    # Valid result — log hint outcome if this was a hint-injected call
                    if last_enqueued_has_hint and trial_logger is not None:
                        _op_lower = (suggested_next_op or "").strip().lower()
                        _outcome  = "none" if _op_lower == "none" else "valid"
                        trial_logger.log_hint_outcome(last_enqueued_type or "llm", _outcome)
                    last_enqueued_has_hint = False

            if step_tracker is not None:
                # Skip VLM predictions until the first LLM call has initialized the
                # tracker — the first VLM fires before any assembly and may guess a
                # late step from a visible object (e.g. float_bowl → step 5).
                is_vlm_result = "current_action" in llm_plan
                if not is_vlm_result or step_tracker._has_evidence:
                    stage = llm_plan.get("stage of assembly",
                                          llm_plan.get("stage_of_assembly", ""))
                    predicted_step = step_tracker.resolve_step_number(stage)
                    if predicted_step:
                        step_tracker.record_prediction(predicted_step)
            if llm_result_str != last_recorded_result:
                last_recorded_result = llm_result_str

            if suggested_next_op == next_task_candidate:
                candidate_confirmation_count += 1
            else:
                next_task_candidate          = suggested_next_op
                candidate_confirmation_count = 1

            effective_threshold = CONFIRMATION_THRESHOLD if first_plan_dispatched else STARTUP_CONFIRMATION
            if (candidate_confirmation_count >= effective_threshold
                    and confirmed_task_plan is None):
                confirmed_task_plan = llm_plan
                # Track which call_id confirmed this plan for dispatch logging.
                if trial_logger is not None:
                    last_confirmed_call_id = trial_logger._call_counter
                _confirmed_op = (confirmed_task_plan.get("next operation")
                                 or confirmed_task_plan.get("next_operation") or "?")
                print(f"[Main] *** Task confirmed: '{_confirmed_op}' ***")
                next_task_candidate          = None
                candidate_confirmation_count = 0

        except queue.Empty:
            pass
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            print(f"[Main Loop] Error processing LLM result: {e}")

        # Robot dispatch: send confirmed plan when robot is ready and the buffer
        # has enough fresh HomePose frames for reliable object detection.
        # LLM runs in parallel during robot execution (proactive); the small wait
        # here (~167ms) only applies to the detection snapshot, not to planning.
        if (robot_is_ready and confirmed_task_plan
                and len(robot_frame_buffer) >= ROBOT_BUFFER_MIN_DISPATCH):
            print(f"[Main] Dispatching plan to robot worker ({len(robot_frame_buffer)} fresh frames).")
            with state_lock:
                shared_state["robot_is_ready"] = False

            _dispatching_plan = confirmed_task_plan
            robot_command_queue.put({"plan": _dispatching_plan,
                                     "robot_frames": list(robot_frame_buffer),
                                     "robot_frame":  frame_robot.copy()})
            if trial_logger is not None and last_confirmed_call_id is not None:
                trial_logger.log_dispatch(
                    dispatched_from_call_id = last_confirmed_call_id,
                    next_operation = str(_dispatching_plan.get(
                        "next_operation", _dispatching_plan.get("next operation", ""))),
                )
            if step_tracker is not None:
                _stage = _dispatching_plan.get(
                    "stage of assembly", _dispatching_plan.get("stage_of_assembly", ""))
                _dispatch_step = step_tracker.resolve_step_number(_stage)
                if _dispatch_step:
                    step_tracker.record_dispatch(_dispatch_step)
                elif not step_tracker._has_evidence:
                    # First dispatch with unresolvable stage (e.g. VLM 'idle'):
                    # anchor next_op as the new current step. Without this the
                    # tracker stays uniform until the visibility update fires,
                    # which then misreads remote-zone absences as "step 1 done".
                    _next_op = _dispatching_plan.get(
                        "next operation", _dispatching_plan.get("next_operation", ""))
                    _next_op_step = step_tracker.resolve_step_number(_next_op)
                    if _next_op_step:
                        step_tracker.record_first_dispatch(_next_op_step)

            confirmed_task_plan    = None
            first_plan_dispatched  = True
            last_submitted_action  = None
            last_llm_result        = None
            last_recorded_result   = None
            pending_augmented_hint = None   # stale hint no longer applies after dispatch
            # Drop stale results so an in-flight inference cannot pollute the next step.
            try:
                while True:
                    llm_result_queue.get_nowait()
            except queue.Empty:
                pass
            next_task_candidate          = None
            candidate_confirmation_count = 0
            data_buffer.clear()            # discard frames from before/during dispatch; fresh start

        # Fire quickly after robot returns home; wait longer while executing to
        # skip the grasping phase and observe real assembly gestures.
        frame_threshold = MIN_FRAMES_FOR_INFERENCE if robot_is_ready else MIN_FRAMES_DURING_ROBOT
        enough_frames   = len(data_buffer) >= frame_threshold

        # Last-step gate: suppress LLM/VLM once the tracker believes the last step
        # is current AND at least one robot dispatch has happened.
        # - "most likely == num_steps" replaces the N-1 completion-threshold check:
        #   visibility evidence spreads equal VISIBILITY_BOOST across all "next step"
        #   slots, so no individual P(done) reaches 90% — they tie at ~25% each.
        # - "first_plan_dispatched" ensures inference is not suppressed before the
        #   robot has been sent for the last step's objects, which lets the system
        #   start correctly even when the operator begins from the last step.
        if step_tracker is not None and step_tracker._has_evidence:
            _sp = step_tracker.get_step_probabilities()
            last_step_only = (max(_sp, key=_sp.get) == step_tracker.num_steps
                              and first_plan_dispatched)
        else:
            last_step_only = False
        if last_step_only and not inference_suppressed:
            print("[Main] Last step reached — suppressing further LLM/VLM inference.")
            inference_suppressed = True
        elif not last_step_only and inference_suppressed:
            inference_suppressed = False

        if (llm_task_queue.empty()
                and not llm_inference_busy.is_set()
                and confirmed_task_plan is None
                and not last_step_only
                and not workspace_clear
                and (enough_frames or (robot_just_finished and len(data_buffer) > 0))):
            # Snapshot buffer once to avoid repeated deque-to-list conversions.
            buf_list = list(data_buffer)

            # Counter/dedup on smoothed_action (pre-delta): delta suffixes vary per-frame
            # and prevent majority voting. The augmented current_action is sent to the LLM.
            smoothed_actions = [d.get("smoothed_action", d.get("current_action", "nothing")) for _, d in buf_list]
            most_common_smoothed = Counter(smoothed_actions).most_common(1)[0][0]

            # Prefer the most recent frames for assembly detection: once the
            # operator is clearly assembling, early-buffer "nothing" frames
            # from before/between gestures should not outvote the current state.
            # Use last-10-frame window; fire LLM if ≥30% show is_assembly=1.
            recent_window = buf_list[-10:]
            assembly_count_recent = sum(d.get("is_assembly", 0) for _, d in recent_window)
            most_common_is_assembly = 1 if assembly_count_recent >= max(1, len(recent_window) // 3) else 0

            # Most recent augmented action matching the winning smoothed base → freshest delta.
            most_common_action = next(
                (d.get("current_action", most_common_smoothed)
                 for _, d in reversed(buf_list)
                 if d.get("smoothed_action", d.get("current_action", "")) == most_common_smoothed),
                most_common_smoothed,
            )

            context          = step_tracker.completion_summary_for_prompt() if step_tracker is not None else ""
            step_probs_raw   = _snapshot_step_probs(step_tracker)

            if most_common_is_assembly == 0:
                if most_common_smoothed == last_submitted_action and last_llm_result is not None:
                    print(f"[Main] Scene unchanged — replaying cached VLM result.")
                    if llm_result_queue.empty():
                        llm_result_queue.put(last_llm_result)
                elif (not robot_is_ready
                      and (time.time() - last_vlm_fire_time) < VLM_MIN_INTERVAL_DURING_ROBOT):
                    # Throttle VLM while robot is executing; rapid non-assembly calls
                    # are usually operator-pause artefacts with unstable outputs.
                    print(f"[Main] VLM skipped — cooldown active (robot busy).")
                else:
                    print(f"[Main] Non-assembly detected ('{most_common_smoothed}') — triggering VLM...")
                    last_submitted_action = most_common_smoothed
                    last_vlm_fire_time    = time.time()
                    buf_len = len(buf_list)
                    _vlm_raw = [
                        buf_list[0][0],
                        buf_list[max(1, buf_len // 3)][0],
                        buf_list[max(1, 2 * buf_len // 3)][0],
                        buf_list[-1][0],
                    ]
                    vlm_frames = []
                    _frame_detections: list[list[str]] = []
                    for _f in _vlm_raw:
                        _r = perception.object_model(_f, verbose=False)
                        if _r and _r[0].boxes and len(_r[0].boxes):
                            _frame_detections.append(
                                [_r[0].names[int(_c)] for _c in _r[0].boxes.cls.tolist()])
                            vlm_frames.append(_r[0].plot(masks=False, conf=False, line_width=2))
                        else:
                            _frame_detections.append([])
                            vlm_frames.append(_f)
                    vlm_task = {"type": "vlm",
                                "data": vlm_frames,
                                "frame_detections": _frame_detections,
                                "step_completion_context": context,
                                "step_probs": step_probs_raw}
                    if pending_augmented_hint:
                        vlm_task["augmented_hint"] = pending_augmented_hint
                        pending_augmented_hint = None
                    last_enqueued_type     = "vlm"
                    last_enqueued_context  = context
                    last_enqueued_has_hint = "augmented_hint" in vlm_task
                    llm_inference_busy.set()
                    llm_task_queue.put(vlm_task)
                    perception.consume_delta()
            else:
                if most_common_smoothed == last_submitted_action and last_llm_result is not None:
                    print(f"[Main] Scene unchanged — replaying cached result.")
                    if llm_result_queue.empty():
                        llm_result_queue.put(last_llm_result)
                else:
                    print(f"[Main] Assembly detected ('{most_common_action}') — triggering LLM...")
                    last_submitted_action = most_common_smoothed
                    llm_task = {"type": "llm",
                                "data": {"semantic_action": most_common_action},
                                "step_completion_context": context,
                                "step_probs": step_probs_raw}
                    if pending_augmented_hint:
                        llm_task["augmented_hint"] = pending_augmented_hint
                        pending_augmented_hint = None
                    last_enqueued_type     = "llm"
                    last_enqueued_context  = context
                    last_enqueued_has_hint = "augmented_hint" in llm_task
                    llm_inference_busy.set()
                    llm_task_queue.put(llm_task)
                    perception.consume_delta()

            data_buffer.clear()

        # Robot camera display.
        robot_display = frame_robot.copy()
        for det in last_robot_detections:
            x1, y1, x2, y2 = map(int, det["coords"])
            cls  = det["class"]
            conf = det.get("confidence", 0.0)
            col  = (int(hash(cls) % 150 + 80),
                    int((hash(cls) >> 4) % 100 + 155),
                    int((hash(cls) >> 8) % 150 + 80))
            cv2.rectangle(robot_display, (x1, y1), (x2, y2), col, 2)
            cv2.putText(robot_display, f"{cls} {conf:.2f}",
                        (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, col, 2, cv2.LINE_AA)

        robot_display_small = cv2.resize(robot_display, (ROBOT_DISP_W, ROBOT_DISP_H))

        if step_tracker is not None:
            comp_probs  = step_tracker.get_completion_probabilities()
            step_probs  = step_tracker.get_step_probabilities()
            has_ev      = step_tracker._has_evidence
            most_likely = max(step_probs, key=step_probs.get) if has_ev else None
            for i in range(step_tracker.num_steps):
                s          = i + 1
                p_current  = step_probs[s]
                p_done     = comp_probs[s]
                is_current = has_ev and (s == most_likely)
                is_done    = has_ev and p_done >= step_tracker.completion_threshold
                color = (0, 200, 0) if is_done else (
                    (0, 200, 255) if is_current else (180, 180, 180))
                display_pct = p_done if is_done else p_current
                bar_w = int(display_pct * 120)
                y = annotated_perception.shape[0] - 14 - (step_tracker.num_steps - i) * 18
                cv2.rectangle(annotated_perception, (8, y), (8 + bar_w, y + 14), color, -1)
                tag = "DONE" if is_done else ("NOW" if is_current else "")
                cv2.putText(annotated_perception, f"S{s} {display_pct:.0%} {tag}",
                            (135, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # Trial status HUD
        if trial_logger is not None:
            if trial_logger.trial_running:
                elapsed   = time.time() - trial_logger._start_time
                hud_text  = f"RECORDING  {elapsed:.0f}s  [ESC=stop]"
                hud_color = (0, 0, 220)    # red
            else:
                hud_text  = f"TRIAL DONE  ATCT={trial_logger.atct:.0f}s  [ESC=exit]"
                hud_color = (0, 200, 0)    # green
            cv2.putText(robot_display_small, hud_text,
                        (10, robot_display_small.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2, cv2.LINE_AA)

        cv2.imshow("Assembly — Robot Camera", robot_display_small)
        cv2.imshow("Assembly — Perception",   annotated_perception)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            if trial_logger is not None and trial_logger.trial_running:
                trial_logger.stop_trial()
            break


def run_disassembly_loop(pipeline_robot, state_lock, shared_state,
                         stabilize_seconds=3.0):
    """
    DISASSEMBLY LOOP — robot clears objects from shared to remote zone.

    Periodically snapshots the robot camera and dispatches a clear-shared
    command whenever the robot is idle.  Press ESC to trigger a final
    container-return pass before stopping.
    """
    ready_since        = None
    cleanup_requested  = False
    cleanup_dispatched = False

    print("\n[Disassembly] Loop started. Press ESC to stop (containers will be returned to remote).")
    while True:
        frames_r      = pipeline_robot.wait_for_frames()
        color_frame_r = frames_r.get_color_frame()
        if not color_frame_r:
            continue

        frame_robot = np.asanyarray(color_frame_r.get_data())
        now = time.time()

        with state_lock:
            robot_ready = shared_state["robot_is_ready"]

        if cv2.waitKey(1) & 0xFF == 27 and not cleanup_requested:
            cleanup_requested = True
            print("[Disassembly] Stop requested — waiting for robot, "
                  "then returning containers to remote...")

        # Cleanup path (ESC pressed).
        if cleanup_requested:
            if robot_ready and not cleanup_dispatched:
                pipeline_robot.wait_for_frames()          # discard buffered
                snap       = pipeline_robot.wait_for_frames()
                snap_color = snap.get_color_frame()
                dispatch_frame = (np.asanyarray(snap_color.get_data())
                                  if snap_color else frame_robot)

                plan = {"next_operation": "final_cleanup", "objects_required": []}
                with state_lock:
                    shared_state["robot_is_ready"] = False
                robot_command_queue.put({"plan": plan, "robot_frame": dispatch_frame})
                print("[Disassembly] Dispatching container cleanup command.")
                cleanup_dispatched = True
            elif cleanup_dispatched and robot_ready:
                print("[Disassembly] Cleanup complete. Stopping.")
                break

        # Normal dispatch path.
        elif robot_ready:
            if ready_since is None:
                ready_since = now
            elif (now - ready_since) >= stabilize_seconds:
                pipeline_robot.wait_for_frames()   # discard buffered
                snap         = pipeline_robot.wait_for_frames()
                snap_color   = snap.get_color_frame()
                dispatch_frame = (np.asanyarray(snap_color.get_data())
                                  if snap_color else frame_robot)

                plan = {"next_operation": "disassembly", "objects_required": []}
                with state_lock:
                    shared_state["robot_is_ready"] = False
                robot_command_queue.put({"plan": plan, "robot_frame": dispatch_frame})
                print("[Disassembly] Dispatching clear-shared command.")
                ready_since = None
        else:
            ready_since = None

        display = frame_robot.copy()
        if cleanup_requested and not cleanup_dispatched:
            label = "WAITING FOR ROBOT — cleanup pending..."
            color = (0, 165, 255)
        elif cleanup_requested and cleanup_dispatched:
            label = "RETURNING CONTAINERS TO REMOTE..."
            color = (0, 165, 255)
        elif robot_ready and ready_since is not None:
            wait_left = max(0.0, stabilize_seconds - (now - ready_since))
            label = f"READY — dispatching in {wait_left:.1f}s"
            color = (0, 255, 0)
        elif robot_ready:
            label = "READY"
            color = (0, 255, 0)
        else:
            label = "ROBOT WORKING..."
            color = (0, 165, 255)
        cv2.putText(display, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(display, "ESC to stop", (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Disassembly Mode — Robot Camera", display)


def main():
    parser = argparse.ArgumentParser(description="Assembly Task Planner and Robot Controller")
    parser.add_argument("--perception-serial", type=str,
                        default="327122078157",
                        help="RealSense serial number for the perception camera")
    parser.add_argument("--robot-serial", type=str,
                        default="207122078463",
                        help="RealSense serial number for the robot camera")
    parser.add_argument("--force-recalibrate", action="store_true", 
                        help="Force recalibration of the workspace homography")
    parser.add_argument("--silent-setup", action="store_true",
                        help="Skip the GP setup confirmation window")
    parser.add_argument("--disassembly", action="store_true",
                        help="Disassembly mode: robot clears objects from shared to remote zone. "
                             "Only the robot camera is used; no LLM inference.")
    parser.add_argument("--stabilize-seconds", type=float, default=6.0,
                        help="Seconds to wait after robot is ready before dispatching a clearing "
                             "command (default: 6.0). Only used with --learn-and-clear.")
    parser.add_argument("--with-context", action="store_true",
                        help="Enable step completion tracking context for LLM/VLM. "
                             "Tracks which steps are likely done and narrows predictions.")
    parser.add_argument("--with-filter", action="store_true",
                        help="Enable scene-consistency filter on LLM/VLM predictions. "
                             "Validates objects_required against YOLO-confirmed scene "
                             "before dispatch; injects augmented hint on next trigger.")
    parser.add_argument("--local-model", action="store_true",
                        help="Use the local OpenVINO model (infer_LLM) instead of the "
                             "Hugging Face API (infer_LLM_HF, default).")
    parser.add_argument("--participant-id", type=str, default=None,
                        help="Participant identifier for trial logging (e.g. P001). "
                             "If omitted, trial logging is disabled.")
    parser.add_argument("--trial-number", type=int, default=1,
                        help="Trial number for this participant (1-based)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory to save trial logs (default: logs/)")
    args = parser.parse_args()

    trial_logger = None
    if args.participant_id is not None and not args.disassembly:
        from trial_logger import TrialLogger
        # Condition label encodes which experimental branch this run belongs to.
        if args.with_filter and args.with_context:
            _condition = "C_with_filter_and_context"
        elif args.with_filter:
            _condition = "C_with_filter"
        elif args.with_context:
            _condition = "B_with_context"
        else:
            _condition = "B_no_context"
        trial_logger = TrialLogger(
            participant_id = args.participant_id,
            condition      = _condition,
            trial_number   = args.trial_number,
            log_dir        = args.log_dir,
            llm_backend    = "local" if args.local_model else "hf_api",
            filter_enabled = args.with_filter,
        )

    # STEP 1: enumerate cameras.
    rs_ctx  = rs.context()
    devices = rs_ctx.query_devices()

    min_cameras = 1 if args.disassembly else 2
    if len(devices) < min_cameras:
        print(f"\n[Error] Found {len(devices)} devices. Need at least {min_cameras} RealSense camera(s).")
        return

    available_serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
    print(f"[Setup] Available RealSense devices: {available_serials}")

    robot_serial      = args.robot_serial if args.robot_serial else available_serials[0]
    perception_serial = args.perception_serial if args.perception_serial else available_serials[1]

    if not args.disassembly and perception_serial not in available_serials:
        print(f"[Error] Requested Perception Camera ({perception_serial}) not found!")
        return
    if robot_serial not in available_serials:
        print(f"[Error] Requested Robot Camera ({robot_serial}) not found!")
        return

    if args.disassembly:
        print(f"[Setup] Mode            : DISASSEMBLY")
        print(f"[Setup] Robot Camera    : {robot_serial}")
    else:
        print(f"[Setup] Mode            : ASSEMBLY")
        print(f"[Setup] Perception Camera : {perception_serial}")
        print(f"[Setup] Robot Camera      : {robot_serial}")

    # STEP 2: connect to robot + GP setup.  Real: 192.168.125.1:1025  Sim: 127.0.0.1:5000
    print("[Setup] Connecting to robot controller...")
    robot_client = RobotSocketClient(host="192.168.125.1", port=1025)
    robot_client.connect()
    print("[Setup] Moving robot to HOME_OBS before setup...")
    robot_client.return_home()

    print("\n[Setup] Starting GP calibration check...")
    grasp_ctx = GraspContext(robot_serial=robot_serial)
    if args.silent_setup:
        grasp_ctx.setup_silent()
    else:
        grasp_ctx.setup(force_recalibrate=args.force_recalibrate)

    print(f"[Setup] GP setup complete — {len(grasp_ctx.gp_data)} points registered.\n")

    # STEP 3: load perception module and LLM/VLM planner.
    print("\n[Setup] Loading Perception Module...")
    perception = PerceptionModule()

    if args.disassembly:
        planner = None
        print("[Setup] Disassembly mode — skipping LLM Planner.")
    else:
        print("\n[Setup] Loading LLM Planner...")
        planner = LLM_planner(load_local_model=args.local_model)

    # STEP 4: start background threads.
    if not args.disassembly:
        llm_thread = threading.Thread(
            target=llm_inference_worker,
            args=(planner, args.local_model, trial_logger),
            daemon=True)
        llm_thread.start()

    robot_thread = threading.Thread(
        target=robot_command_worker,
        args=(perception, state_lock, grasp_ctx, robot_client),
        daemon=True
    )
    robot_thread.start()

    # STEP 5: open camera pipelines.
    pipeline_robot = rs.pipeline()
    config_robot   = rs.config()
    config_robot.enable_device(robot_serial)
    config_robot.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline_perception = None
    if not args.disassembly:
        pipeline_perception = rs.pipeline()
        config_perception   = rs.config()
        config_perception.enable_device(perception_serial)
        config_perception.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    print("[Setup] Starting robot camera stream...")
    try:
        pipeline_robot.start(config_robot)
        if pipeline_perception is not None:
            pipeline_perception.start(config_perception)
            print("[Setup] Starting perception camera stream...")

        if args.disassembly:
            run_disassembly_loop(pipeline_robot, state_lock, shared_state,
                                 stabilize_seconds=3.0)
        else:
            step_tracker = StepTracker(planner.memory) if args.with_context else None
            if trial_logger is not None:
                trial_logger.start_trial()
            run_assembly_loop(
                pipeline_perception, pipeline_robot, perception,
                robot_client, grasp_ctx, state_lock, shared_state,
                step_tracker=step_tracker,
                trial_logger=trial_logger,
                filter_enabled=args.with_filter,
                assembly_memory=planner.memory)

    finally:
        with state_lock:
            llm_calls = shared_state["llm_call_count"]
            vlm_calls = shared_state["vlm_call_count"]
            llm_times = list(shared_state["llm_times"])
            vlm_times = list(shared_state["vlm_times"])

        print(f"\n--- Run Summary ---")
        if not args.disassembly:
            llm_mean = sum(llm_times) / len(llm_times) if llm_times else 0.0
            vlm_mean = sum(vlm_times) / len(vlm_times) if vlm_times else 0.0
            print(f"LLM calls : {llm_calls:>4}   avg {llm_mean:.1f}s")
            print(f"VLM calls : {vlm_calls:>4}   avg {vlm_mean:.1f}s")
        print(f"-------------------")

        if not args.disassembly and trial_logger is not None:
            total_steps = len(planner.memory) if planner is not None else None
            while True:
                prompt = (f"\n[TrialLogger] Correct steps ( _/{total_steps} ): "
                          if total_steps is not None else
                          "\n[TrialLogger] Correct steps: ")
                raw = input(prompt).strip()
                try:
                    correct = int(raw)
                    if total_steps is not None and not (0 <= correct <= total_steps):
                        print(f"  Please enter a number between 0 and {total_steps}.")
                        continue
                    break
                except ValueError:
                    print("  Invalid input — please enter an integer.")
            execution_score = round(correct / total_steps, 4) if total_steps else None
            trial_logger.save_summary(execution_score=execution_score)
            trial_logger.close()

        print("Stopping streams...")
        try:
            pipeline_robot.stop()
            if pipeline_perception is not None:
                pipeline_perception.stop()
        except RuntimeError:
            pass
        robot_client.close()
        if perception:
            perception.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()