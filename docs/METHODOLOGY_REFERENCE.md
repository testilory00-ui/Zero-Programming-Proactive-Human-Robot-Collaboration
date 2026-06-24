# Methodology Chapter — Code Reference

This document maps every section of Chapter 2 **Methodology** (from `table_of _context__methodology.pdf`) to the exact code that implements it. Use it to reconstruct each subsection accurately.

---

## 2.1 System Overview

### 2.1.1 Task Description and Design Goals

**The task**: carburetor assembly — 5 sequential steps defined in `learned_memory.json`.  
Each step has: `step number` (int), `step description` (string), `objects_required` (list of YOLO class names).  
The five components are: diaphragm, spring, cover (with screws + screwdriver), throttle stop, float bowl — all mounted onto the carburetor body.

YOLO class names used throughout the system: `carburetor_body`, `diaphragm`, `spring`, `cover`, `throttle_stop`, `float_bowl`, `screw`, `screwdriver`.

**Design goals** (visible from the CLI flags and code structure):
- **Proactivity**: robot prepares the next component *before* the operator asks — triggered by perception of the current action.
- **Safety**: robot only moves after multiple consistent LLM predictions; never interrupts an active operator gesture.
- **Generality / one-shot learning**: a new task is learned from a single operator demonstration with no manual annotation (`learn_procedure.py`).
- **Offline-capable**: primary inference uses a local quantized LLM (Qwen 3 4B INT4 via OpenVINO on GPU); VLM (Gemini) used as fallback or for non-assembly scenes.
- **Modularity**: each concern (perception, planning, robot execution) is isolated in its own module and thread.

**Hardware**:
- Two Intel RealSense cameras:
  - *Perception camera*: 640×480 @ 30 fps — monitors operator hands and workspace (`perception_serial`, `main.py:1065`).
  - *Robot camera*: 1280×720 @ 30 fps — overlooks the workspace from the robot side; used for object detection before pick-and-place (`robot_serial`, `main.py:1068`).
- **ABB GoFa** collaborative robot: TCP connection to `192.168.125.1:1025` (real) or `127.0.0.1:5000` (RobotStudio sim).
- **Intel Arc GPU**: runs OpenVINO-compiled Qwen model (`llm.py:25`).

### 2.1.2 Architectural Overview and Design Rationale

The system is a **multi-threaded producer-consumer pipeline** with two operational phases:

- **Learning phase** (`learn_procedure.py`): run once per new task. Operator demonstrates the assembly; the system records and structures the steps autonomously.
- **Inference phase** (`main.py`): run each session. Dual-camera loop that perceives, plans, and dispatches robot commands in real time.

**Thread architecture** (inference phase):

```
Main thread (30 fps)
  RealSense Cameras (2×)
    → PerceptionModule (MediaPipe + YOLO)
    → data_buffer / robot_frame_buffer
    → [reads llm_result_queue]
    → [writes llm_task_queue]
    → [writes robot_command_queue]

LLM Worker Thread  (blocks on llm_task_queue)
  → Qwen 3 4B INT4 / OpenVINO (local) — LLM path
  → Gemini gemini-2.5-flash-lite (API)  — VLM path
  → writes llm_result_queue

Robot Worker Thread  (blocks on robot_command_queue)
  → GraspContext (GP resolution)
  → PerceptionModule.detect_objects_in_frames()
  → RobotSocketClient (TCP)
  → ABB GoFa
```

**Three queues** decouple the fast perception loop from slow inference (`main.py:133-135`):
- `llm_task_queue` (maxsize=1)
- `llm_result_queue` (maxsize=1)
- `robot_command_queue` (maxsize=1)

`llm_inference_busy` (threading.Event, `main.py:139`) prevents double-submission while the worker has dequeued but not yet finished.

`state_lock` (threading.Lock, `main.py:148`) guards `robot_is_ready`, call counters, and timing stats.

**Design rationale** for key choices:
- **Separate perception and robot cameras**: the robot arm occludes the workspace during motion; the perception camera (side view) is unaffected by arm movement.
- **LLM for assembly / VLM for idle**: during active manipulation frames are blurry and hands occlude objects — the compact semantic action string is more reliable input than images. VLM (with images) is used when hands are idle and scene state needs to be read visually.
- **Confirmation threshold**: a single LLM prediction is not dispatched immediately; N identical consecutive predictions are required to prevent noisy single outputs from triggering robot movement.
- **Background threads**: LLM inference takes 5–20 s; running it in a daemon thread ensures the 30 fps perception loop is never blocked.

**Startup sequence** (`main.py:1120-1225`):
1. Enumerate RealSense devices (≥2 required).
2. Connect robot → `return_home()` → `GraspContext.setup()` (ArUco calibration check).
3. Load `PerceptionModule` + `LLM_planner` (slowest step: 30–60 s on first run, OpenVINO GPU compilation).
4. Start LLM worker thread + robot worker thread (daemons).
5. Open camera pipelines → enter main loop.

### 2.1.3 YOLO Fine-Tuning for Object Detection and Segmentation

**Model**: YOLOv8-seg fine-tuned on the carburetor assembly objects. Model file: `best_3.pt`. Loaded in `PerceptionModule.__init__` (`perception.py:18`).

**Classes** (8): `carburetor_body`, `diaphragm`, `spring`, `cover`, `throttle_stop`, `float_bowl`, `screw`, `screwdriver`.

**Two inference modes** in `perception.py`:

1. **Stateful tracking** (`process_yolo`, `perception.py:123-192`):
   - Runs every `YOLO_FREQ=3` frames for performance.
   - Uses `model.track(persist=True)` — assigns persistent IDs across frames.
   - Confirmation: object must appear for `MIN_FRAMES_CONFIRM=8` frames before being added to `confirmed_objects`.
   - Pruning: objects absent for `MAX_ABSENT_FRAMES=30` frames are removed.
   - **ID recovery** (`perception.py:155-165`): if a tracker ID is lost (e.g. occlusion), transfers identity to the first unmatched confirmed object of the same class — prevents phantom re-entries in the inventory delta.
   - Returns `latest_objects`: list of confirmed objects with last-known position (persists through brief occlusion).

2. **Stateless single-frame detection** (`detect_objects_in_frame`, `perception.py:199-217`):
   - No tracking, no ID assignment. Used by the robot worker on a single HomePose frame.
   - Returns list of `{class, coords, confidence}`.

3. **Multi-frame aggregation** (`detect_objects_in_frames`, `perception.py:219-273`):
   - Runs stateless detection on each frame, then clusters detections by class + center proximity (100 px match distance).
   - Returns only objects seen in ≥ `min_frame_count` frames (default 3; set to 2 in robot worker for small objects like spring).
   - Result: `{class, coords (mean bbox), confidence (max)}` — more robust than a single frame for robot dispatch.

Thread safety: `_yolo_lock` (threading.Lock, `perception.py:21`) — the YOLO model is not thread-safe; the robot worker also calls it concurrently.

---

## 2.2 Learning Phase

### 2.2.1 Learning Pipeline Architecture

`learn_procedure.py` — four sequential phases:

**Phase 0 — Inventory scan** (`phase0_inventory_scan`, `learn_procedure.py:201-268`):
- Operator places all assembly objects visibly on the table.
- `PerceptionModule` runs until SPACE is pressed or 450-frame auto-timeout.
- Builds `global_object_list` (sorted list of all detected class names).
- Captures a YOLO-annotated `reference_frame` used as a visual name-to-appearance map for all subsequent VLM calls — lets Gemini use exact YOLO label names rather than guessing.

**Phase 1 — Autonomous recording** (`phase1_record`, `learn_procedure.py:271-381`):
- Operator performs the assembly while `TriggerDetector` monitors perception output.
- On each trigger: 4 frames extracted (before/mounting/mounting/after), sent to `infer_VLM_learn`, result appended to `raw_buffer`.
- **Crash recovery**: after each trigger, `raw_buffer` checkpointed to `learn_procedure_checkpoint.json` and frames to `learn_frames/` directory. On restart, operator chooses C (continue) or D (discard).

**Phase 2 — VLM cleanup** (`infer_VLM_cleanup` in `llm.py:454`):
- Receives all raw observations + representative frames.
- Returns deduplicated, ordered, normalized steps.
- If operator rejects: feedback text appended to prompt for a revision pass (iterative loop).

**Phase 3 — Operator validation** (`phase3_validate` in `learn_utils.py`):
- Operator chooses: Y (approve and save), R (re-run cleanup with feedback), N (restart recording from scratch), X (abort session).

### 2.2.2 Event-Driven Observation and VLM-Based Step Inference

#### TriggerDetector (`learn_procedure.py:49-198`)

State machine: `IDLE → ACTIVE → CHECK_DISPLACEMENT → COOLDOWN`

Uses **instantaneous** hand state (not smoothed, `log_entry["instantaneous_hand_state"]`) to detect state transitions precisely.

State-level hierarchy: `nothing=0`, `pinch=1`, `assembly=2`.

**Transitions**:

- **IDLE → ACTIVE**: hand level ≥ 1. Snapshots all object centroids at this moment (`pre_action_centroids`).
- **ACTIVE** (monitoring):
  - Tracks `peak_level` (highest level reached).
  - `consecutive_below` counter: increments when current level < peak.
  - After `release_confirm_frames=3` consecutive below-peak frames AND duration ≥ `min_active_duration_ms=300 ms`: → `CHECK_DISPLACEMENT`.
  - If duration < 300 ms: → back to IDLE (too brief, discarded).
  - **Fallback**: if duration ≥ `max_active_duration_ms` (default 20 s): trigger fires unconditionally → COOLDOWN.
- **CHECK_DISPLACEMENT** (grace window of 15 frames):
  - Each frame: compares current object centroids vs `pre_action_centroids`.
  - **Displacement detected** (object moved > `displacement_threshold_px=40 px` OR object disappeared): → COOLDOWN, trigger returns `True`.
  - No displacement after 15 frames: → IDLE (action discarded — no physical change occurred).
- **COOLDOWN**: ignores all input for `cooldown_frames=90` frames, then → IDLE.

**Displacement check purpose**: prevents false step recordings from brief accidental hand contact that caused no physical change.

#### Frame extraction and VLM call

Rolling buffer `active_frame_buffer` (max 60 frames ≈ 2 s) accumulates frames during ACTIVE and CHECK_DISPLACEMENT states.

On trigger (`learn_procedure.py:320-349`):
1. `extract_action_frames(active_frame_buffer)` → `(start_frame, mid1, mid2)` evenly spaced from the buffer.
2. `capture_frames(pipeline, n_frames=1)` → `end_frame` (captured after trigger, shows final state).
3. `learn_frames = [start_frame, mid1, mid2, end_frame]` — 4 frames sent to `infer_VLM_learn`.
4. `cleanup_frames = [mid1, mid2]` — mid-action frames stored for Phase 2 cross-referencing.

`infer_VLM_learn` (`llm.py:302-368`):
- Sends reference_frame first (YOLO-annotated, visual name anchor), then the 4 action frames.
- Prompt asks: describe the action in one sentence, identify objects using reference image label names.
- Critical rule: if screws or screwdriver appear in any frame, they MUST be included.
- Returns `{description: str, objects_required: [str]}`.
- Uses `gemini-3.1-flash-lite-preview`.

### 2.2.3 Post-Hoc Cleanup and Operator Validation

`infer_VLM_cleanup` (`llm.py:454-581`):

Input: raw observations (text descriptions + object lists) + up to 12 evenly-spaced representative frames (capped to limit token usage) + `global_object_list`.

**Five-step prompt pipeline**:
1. **Filter**: discard observations where description is vague/empty AND images show no assembly gesture.
2. **Deduplicate**: consecutive observations describing the same physical action → merged into one.
3. **Normalize**: lowercase, underscores for spaces. Use inventory names when they match.
4. **Write steps**: each step = one concise sentence describing one distinct physical action.
5. **Coverage check**: every object in the inventory must appear in ≥1 step's `objects_required`. If missing, find a matching step from images or add a new step at the logical position.

Critical rule: screwdriver in `objects_required` always requires the corresponding fastener in the same step.

**Revision mode**: if `previous_result` and `operator_feedback` are provided, the previous JSON result and the operator's correction text are appended to the prompt. The model applies only the stated correction, keeping everything else.

Uses `gemini-3-flash-preview` (more capable model for complex restructuring).

### 2.2.4 Memory Representation

`learned_memory.json` — JSON array, loaded at startup by `LLM_planner.__init__` as `self.memory`:

```json
[
  {
    "step number": 1,
    "step description": "<one sentence: what was done and which component>",
    "objects_required": ["object_A", "object_B"]
  },
  ...
]
```

Used throughout inference as:
- The assembly step list injected into every LLM/VLM prompt.
- The reference for GP object routing in the robot worker.
- The input to `StepTracker` for probabilistic state estimation.

---

## 2.3 Inference Phase: Proactive Execution

### 2.3.1 Main Loop Architecture

`run_assembly_loop` (`main.py:498-965`).

**Key timing constants** (`main.py:513-517`):
- `CONFIRMATION_THRESHOLD = 1` — predictions needed to confirm a plan (after first dispatch).
- `STARTUP_CONFIRMATION = 2` — higher threshold before first dispatch (prevents a single noisy VLM result at startup from locking the wrong step).
- `MIN_FRAMES_FOR_INFERENCE = 20` — fast re-trigger after robot returns home (~0.7 s observation).
- `MIN_FRAMES_DURING_ROBOT = 180` — slower trigger while robot executes (~6 s; skips the grasping phase so the system observes real assembly gestures, not robot motion).
- `VLM_MIN_INTERVAL_DURING_ROBOT = 5.0 s` — minimum interval between VLM calls while robot is busy (throttles repeated non-assembly triggers).
- `ROBOT_BUFFER_MIN_DISPATCH = 10` — min fresh HomePose frames before dispatching (~167 ms).

**Per-frame loop** (`main.py:555-965`):

```
1. Capture perception frame + robot frame.
2. PerceptionModule.process_frame(perception_frame)
   → hand detection + YOLO tracking + semantic action
   → append (frame, frame_data) to data_buffer

3. Append robot frame to robot_frame_buffer.
   If robot_just_finished: flush robot_frame_buffer (discard motion frames).

4. At HomePose (startup OR robot_just_finished):
   → detect_objects_in_frame(robot_frame) → last_robot_detections
   → StepTracker.record_object_visibility() if enabled
   → workspace_clear gate: if no objects detected → suppress inference

5. Non-blocking read from llm_result_queue:
   → parse JSON → optional scene-consistency filter → confirmation counting
   → if confirmed: set confirmed_task_plan

6. If robot_is_ready AND confirmed_task_plan AND ≥10 fresh robot frames:
   → robot_is_ready = False
   → put {plan, robot_frames} → robot_command_queue
   → clear confirmed_task_plan, flush queues, clear data_buffer

7. If data_buffer has enough frames AND llm not busy AND no confirmed plan
   AND not suppressed (last-step gate or workspace-clear gate):
   → compute most_common_is_assembly (last-10-frame window, ≥30% threshold)
   → if is_assembly: submit LLM task
   → if not is_assembly: submit VLM task (with 4 sampled frames)
   → dedup: if scene unchanged → replay cached result
   → clear data_buffer
```

**Robot-frame YOLO timing**: detection only runs at HomePose (`robot_just_finished OR not home_pose_capture_done`), not during arm motion — avoids captures where the arm occludes the workspace.

**`robot_just_finished`** (`main.py:576`): `robot_is_ready AND NOT prev_robot_ready` — a one-frame edge trigger when the robot returns home.

**Inference suppression gates**:
- **Last-step gate** (`main.py:792-803`): once StepTracker assigns highest probability to the last step AND at least one dispatch has happened, LLM/VLM inference is suppressed.
- **Workspace-clear gate** (`main.py:602-608`): if YOLO finds no objects after robot returns home, inference is suppressed (assembly complete).

### 2.3.2 Perception Module

`PerceptionModule` (`perception.py`) — fuses hand pose estimation with object detection to produce a semantic action string every frame.

#### Hand detection — MediaPipe

`HandLandmarker` in `LIVE_STREAM` mode (async). `detect_async()` called at each frame; result delivered to `save_hand_result` callback (`perception.py:68`). Landmarks used: 0=Wrist, 4=Thumb tip, 8=Index tip.

#### Hand state classification (`elaborate_state`, `perception.py:93-121`)

- **pinch**: 3D Euclidean distance between thumb tip and index tip (normalized coordinates) < `PINCH_THRESHOLD=0.2`.
- **assembly**: both hands pinching AND normalized wrist-to-wrist distance < `ASSEMBLY_WRIST_THRESHOLD=0.7` — captures the two-handed gesture of joining components.
- **nothing**: otherwise.

#### Semantic action mapping (`calculate_semantic_action`, `perception.py:275-339`)

- **pinch state**: for each pinching hand, computes thumb-index midpoint and finds the nearest YOLO object within `HAND_OBJECT_PROXIMITY=50 px`. Result: `"pinch <class>"` or `"pinch <class1>, <class2>"`.
- **assembly state**: computes wrist midpoint, finds objects within `HAND_OBJECT_PROXIMITY × 4 = 200 px`. Result: `"assembly with <objects>"`.

#### Smoothing (`process_frame`, `perception.py:391-414`)

`SLIDING_WINDOW_MS=300 ms` sliding window with majority vote over `global_state_history` and `semantic_action_history`. Two action strings are maintained:
- `smoothed_action` (pre-delta): used for Counter and dedup (stable across frames).
- `current_action` (with delta suffix): sent to LLM.

#### Inventory delta (`perception.py:341-369`)

Tracks which classes entered/left `confirmed_objects` in the last `INVENTORY_DELTA_WINDOW_MS=1500 ms`.  
Appended as `" | recent: <class> entered"` suffix to `current_action`.  
**Sticky**: delta parts accumulate in `_pending_delta_parts` until `consume_delta()` is explicitly called after LLM dispatch (`main.py:882, 903`). This prevents the delta from clearing during the 5–20 s inference latency window.

**Purpose**: disambiguates steps with identical gesture descriptions (e.g. both step 2 and step 3 involve "assembly with ...") by surfacing which new object just appeared on the table.

#### Output dict (returned by `process_frame`)

| Key | Description | Used for |
|-----|-------------|----------|
| `current_action` | Semantic action with delta suffix | Sent to LLM |
| `smoothed_action` | Pre-delta action | Counter / dedup |
| `is_assembly` | 1 if "assembly" in smoothed action AND ≥2 involved objects | LLM vs VLM routing |
| `confirmed_objects` | Set of currently tracked class names | Scene filter, StepTracker |
| `instantaneous_hand_state` | Per-frame (unsmoothed) hand state | TriggerDetector (learning) |

### 2.3.3 Dual-Model Cognitive Routing

Routing decision in `run_assembly_loop` (`main.py:837-903`):

| Condition | Route | Model | Input |
|-----------|-------|-------|-------|
| `most_common_is_assembly == 1` | LLM path | Qwen 3 4B INT4 (local, OpenVINO) or Llama-4-Scout-17B (HF API) | `semantic_action` string |
| `most_common_is_assembly == 0` | VLM path | Gemini `gemini-2.5-flash-lite` | 4 YOLO-annotated frames + YOLO detection lists |

`most_common_is_assembly` computed from the **last-10-frame window** of `data_buffer`: value is 1 if ≥30% of those frames have `is_assembly=1` (`main.py:823-824`). Recency bias: prefers the current operator state over early-buffer "nothing" frames from between gestures.

Default backend: HuggingFace Inference API (`infer_LLM_HF`). Switch to local with `--local-model` (`infer_LLM`).

#### LLM prompt (`infer_LLM` / `infer_LLM_HF`, `llm.py:61-225`)

Structure:
1. Optional `[SCENE CORRECTION — MUST FOLLOW]` block (from scene-consistency filter).
2. Assembly steps list with `objects_required` per step.
3. Optional step-completion context (StepTracker summary).
4. `OBSERVED ACTION: "<action_line>"` (delta suffix removed from main line).
5. Optional `RECENT CHANGE: <delta>` line (reformatted from delta suffix so smaller models don't treat it as noise — `_split_action_delta`, `llm.py:46-59`).
6. Instructions + guidance (different text when hint is present vs absent).
7. Required JSON output: `{stage of assembly, next operation, objects required}`.

`max_new_tokens=160` (JSON ≈ 50–70 tokens; 160 is a safe cap). Qwen3 `enable_thinking=False` suppresses the `<think>…</think>` block to save latency.

#### VLM prompt (`infer_VLM`, `llm.py:227-300`)

4 frames in chronological order labeled "earliest", "mid-action early", "mid-action late", "latest". Each frame followed immediately by its YOLO detection list (objects reliably detected even when visually occluded in that frame).

Prompt defines:
- `stage_of_assembly`: step the operator is **actively performing** (hands visibly picking/inserting/fastening). Must be "idle" if hands are not performing an assembly gesture — explicitly prevents VLM from confusing object presence on table with ongoing assembly.
- `next_operation`: step the robot should prepare next. Wraps from last step to step 1.

VLM frames are YOLO-annotated (bounding boxes drawn via `results[0].plot()`) before being sent, providing visual anchors for the model (`main.py:864-868`).

Required JSON output: `{current_action, stage_of_assembly, next_operation, objects_required}`.

#### Deduplication (`main.py:838-843, 884-887`)

If `smoothed_action == last_submitted_action` AND a cached result exists: replay the cached result directly to `llm_result_queue` without a new inference call. Prevents redundant LLM/VLM calls when the scene has not changed between triggers.

### 2.3.4 Probabilistic State Estimation

`StepTracker` (`step_tracker.py`) — enabled with `--with-context`.

Maintains a **softmax distribution** over N assembly steps plus one virtual "completed" state, represented as a logit vector of length N+1. Every evidence update first decays all logits by `decay_factor=0.95`, then adds new evidence.

`LOGIT_CAP=5.0` prevents softmax saturation: the dominant step peaks at ≈97%, not 100%, keeping the distribution correctable.

#### Evidence sources

**1. `record_prediction(step)` — LLM/VLM predicted step N** (`step_tracker.py:138-171`):
- Decays all logits × 0.95.
- `logits[N-1] += PREDICTION_BOOST=1.5` (capped at 5.0).
- `logits[N] += PREDICTION_BOOST × FORWARD_BOOST_RATIO=0.08` — small nudge to the next step.
  - **Purpose**: breaks self-reinforcing echo. Without it, repeated identical predictions lock the distribution; the forward boost makes the next step gradually approach dominance even before an explicit signal.
- Regression damping: if `predicted_step < frontier - REGRESSION_GAP=0`, boost multiplied by `REGRESSION_DAMPING=0.15`. Non-zero so correct repeated evidence can still override a wrong frontier.

**2. `record_object_visibility(visible_classes)` — robot-camera YOLO at HomePose** (`step_tracker.py:173-217`):
- **Signature objects** (`_compute_signature_objects`): per-step objects that appear in fewer than ceil(N/2) steps. Shared tools (screwdriver, carburetor_body) are excluded.
- For each step whose signature objects are **all absent** from `visible_classes`: `logits[next_step-1] += VISIBILITY_BOOST=2.0`. (The object was assembled — it's no longer on the table in its raw form.)
- Lowest step still **present** on table: `logits[step-1] += VISIBILITY_PRESENT_BOOST=2.0`.
- No-op if no absence signal exists (to avoid flattening with only presence evidence).
- Called only at HomePose; first call after startup skipped (`skip_first_visibility`, `main.py:594-598`) because most objects are still in the remote zone and their apparent absence is structural, not evidence of completion.

**3. `record_dispatch(current_step)` — robot dispatched** (`step_tracker.py:241-269`):
- Strongest signal (double-confirmed LLM + robot command sent).
- `logits[current_step-1] *= DISPATCH_DECAY=0.30` (logit drops to ≈30%).
- `logits[current_step] += DISPATCH_BOOST=2.0`.
- Effect: next step jumps to ≈94% dominant; current step drops to ≈3% (not zeroed — correctable if dispatch was wrong).

**4. `record_first_dispatch(target_step)`** (`step_tracker.py:219-239`):
- Used when the first dispatch has an unresolvable stage (e.g. VLM stage='idle' at startup).
- Anchors `target_step` without decaying a non-existent "previous" step.

#### Derived probabilities

- `get_step_probabilities()`: P(current=i) for each step via softmax.
- `get_completion_probabilities()`: P(step j done) = Σ P(current=i) for i > j.
  - Monotonicity enforced: P(j done) ≤ P(j-1 done).
  - **Visited cap**: a step that has never been the most-likely current step is capped just below `completion_threshold=0.90`. Prevents a single ahead-of-frontier guess from marking a step DONE before it was ever "current".
- `completed_steps`: steps with P(done) ≥ 0.90.

#### Prompt injection (`completion_summary_for_prompt`, `step_tracker.py:410-475`)

Before first evidence: fixed message "assembly just begun — Step 1 most likely."

After evidence — short directive style:
- Single dominant step: `"STEP TRACKER: the assembly is most likely at Step N (confidence X%)"`.
- Ambiguous top-2 (margin < 15%): both named. If they are consecutive (N, N+1), directs model to prefer the later one — this is the forward-boost deadlock pattern where step N has been echoed enough to lift N+1 to match.
- Completed steps appended: `"Steps X, Y appear already done."`.

`resolve_step_number(description)` (`step_tracker.py:378-405`): matches a free-text LLM output to a step number by content-word overlap. Requires ≥2 shared content words, or 1 word that is unique to exactly one step (`_distinctive_words`, computed at init).

---

## 2.4 Robot Execution Layer

### 2.4.1 Grasp Point Registry and Coordinate Mapping

`GraspContext` (`grasp_context.py`) — single shared object, instantiated once in `main()` and passed read-only to the robot worker thread.

**ArUco homography calibration** (`sheet_and_calibration.py`): ArUco markers on a calibration sheet compute a 3×3 homography `H` mapping robot-camera pixel coordinates → robot workspace coordinates. Saved to `homography.json`. Recomputed only with `--force-recalibrate`.

Setup modes:
- Interactive (`setup()`): opens robot camera, shows GP overlay, waits for operator Q to confirm.
- Silent (`setup_silent()`): loads existing `homography.json` without any window.

**GP data structure** (`grasp_context.py:59`): keyed by GP ID. Each entry: `type`, `zone`, `px_center` (pixel coord in robot camera image), `px_grasp` (grasp point pixel coord), `label`.

**GP types**:
- `standard`: regular object slot, matched by YOLO bbox distance.
- `screwdriver`: dedicated screwdriver slot (1 per zone).
- `screws_container`: dedicated screw container (1 per zone).
- `general_container`: container routed zone-to-zone by paired naming convention: `CTR_GENERAL_R1 ↔ CTR_GENERAL_S1`, `CTR_GENERAL_R2 ↔ CTR_GENERAL_S2` (regex pairing in `find_paired_general_container`, `grasp_context.py:102-117`).

**Zones**: `"shared"` (operator's working area, accessible to both operator and robot) and `"remote"` (robot storage area, out of operator reach).

**GP resolution in `robot_command_worker`** (`main.py:303-478`):

Step 1 — Detect objects: `detect_objects_in_frames(robot_frames, min_frame_count=2)` on up to 20 buffered HomePose frames. Small objects (spring) may appear in only 1–2 frames; `min_frame_count=2` avoids silently dropping them.

Step 2 — Zone assignment: for each detected non-screw object, `find_nearest_gp(bbox_center, zone, tolerance_px=140)`. Objects matching the remote zone → `remote_area`; shared zone → `shared_area`. Screwdriver uses `type_filter="screwdriver"` to target its dedicated GPs.

Step 3 — Action classification:
- `objects_to_bring`: in `remote_area` AND in `objects_required`.
- `objects_to_remove`: in `shared_area` AND NOT in `objects_required`.

Step 4 — GP pair assignment:
- **BRING pass 1** (standard objects): `general_container` GPs use `find_paired_general_container`; standard GPs use nearest free shared GP by pixel distance (`get_free_gps`).
- **BRING pass 2** (screwdriver): `find_screwdriver_gp("shared")`.
- **REMOVE** (shared → remote): symmetric logic.
- **Screws container** (Track B, `main.py:450-478`): routed by GP type (`find_screws_container`), independent of YOLO bbox position. `screws_in_shared` state variable persists across robot calls — container is only moved when the required/not-required status changes.

**Z offsets** (`OBJECT_Z_OFFSETS`, `main.py:125-131`): per-object height corrections above nominal GP plane.
- `carburetor_body`: +70 mm
- `float_bowl`: +20 mm
- `diaphragm`: -3 mm
- `cover`: -1 mm
- `throttle_stop`: +2 mm

`occupied_gp_ids` set prevents double-booking during planning. On batch failure: `screws_in_shared` restored to `pre_batch_screws` (assume nothing moved).

### 2.4.2 Dispatch Filter and Safety Logic

Two independent mechanisms prevent incorrect or premature robot dispatch:

#### Confirmation threshold (`main.py:702-719`)

`next_task_candidate` tracks the most recent `next_operation` string from the LLM/VLM.  
`candidate_confirmation_count` increments when consecutive results agree; resets to 1 on any change.

`effective_threshold`:
- `STARTUP_CONFIRMATION=2` before first dispatch (guards against a single noisy VLM result at startup).
- `CONFIRMATION_THRESHOLD=1` after first dispatch (single confirmed prediction is sufficient once assembly is underway).

When threshold reached AND `confirmed_task_plan is None`: plan is locked.

On dispatch (`main.py:763-777`): all queues flushed, `data_buffer` cleared, `last_submitted_action` and all dedup state reset, `pending_augmented_hint` cleared (stale hint no longer applies after dispatch).

#### Scene-consistency filter (`main.py:619-686`) — enabled with `--with-filter`

After each LLM/VLM result, before the confirmation counter:

1. Runs `detect_objects_in_frames` on the current `robot_frame_buffer` (min_frame_count=1 for fast check).
2. `_scene_filter(plan, confirmed_classes)` (`main.py:101-121`): checks that every non-screw/screwdriver object in `objects_required` is YOLO-confirmed present. Plans with `next_operation = "none"` or `"final_cleanup"` always pass.
3. **If invalid**:
   - Plan rejected (not counted toward confirmation).
   - Builds `augmented_hint` string containing: rejected prediction, missing objects, objects actually on table, compatible steps (steps whose required objects are all present), assembly progress from StepTracker.
   - Stored as `pending_augmented_hint`.
   - `last_submitted_action` reset → forces a new inference on next trigger.
   - `data_buffer` cleared → retry with fresh perception frames.
4. **If valid**: `pending_augmented_hint` cleared; confirmation counter proceeds normally.

`pending_augmented_hint` is injected into the next LLM or VLM task (`main.py:874-879, 895-900`) as a `[SCENE CORRECTION — MUST FOLLOW]` block — so the corrective context is used on the very next inference call without any extra delay.

Skip objects: `_FILTER_SKIP_OBJS = {"screw", "screws", "screwdriver", "carburetor_body"}` — small/always-present objects that YOLO may miss or that are structurally present throughout.

### 2.4.3 RAPID Module and TCP Socket Communication

`RobotSocketClient` (`robot_socket_client.py`).

**Wire protocol** (newline-terminated ASCII):
```
Python → RAPID : "GP_R1,GP_S1,z_offset|GP_R2,GP_S2,z_offset\n"
RAPID  → Python: "OK\n"    (after all motions complete + robot at HomePose)
Python → RAPID : "HOME\n"
RAPID  → Python: "OK\n"
Python → RAPID : "DONE\n"
RAPID  → Python: "BYE\n"   (session end)
```

**Batch splitting** (`execute_batch`, `robot_socket_client.py:84-119`):  
RAPID string cap is 80 chars. Batches are split into ≤79-char chunks (token separator is `|`). Each chunk is sent and confirmed with `"OK"` before the next is sent — RAPID executes motions in sequence within each chunk.

`RECV_TIMEOUT=130 s` (above RAPID's internal 120 s timeout so Python always times out after RAPID, not before).

`close()` calls `socket.shutdown(SHUT_RDWR)` before `close()` — unblocks any thread blocked in `recv()` (the robot worker waiting for "OK") on Ctrl+C / ESC.

`execute_batch_from_commands` (`robot_socket_client.py:129-141`): converts `bring_commands + remove_commands` dicts to `(pick_gp_id, place_gp_id, z_offset)` triples and delegates to `execute_batch`. If the combined list is empty, sends `HOME` instead.

**RAPID project**: `RobotStudio/pick_and_place.modx` + `TargetPoint_Robot.modx`.  
Real robot: `192.168.125.1:1025`. RobotStudio simulation: `127.0.0.1:5000` (port-forwarded in station config).

---

## Key Design Decisions Worth Explaining in the Chapter

1. **Dual-path routing** (LLM vs VLM by hand state): during active manipulation, camera frames are blurry and hands occlude objects — the LLM receives the compact semantic action string instead of images. The VLM is invoked when hands are idle and scene state must be read visually.

2. **Startup confirmation = 2 / runtime confirmation = 1**: the first VLM call fires before any assembly context is established and can hallucinate a late step from a visible object (e.g. float_bowl → step 5). A higher startup threshold absorbs one wrong prediction before locking a plan.

3. **Forward-boost in StepTracker**: prevents the self-reinforcing echo where ambiguous actions cause the LLM to echo the same step indefinitely. Even a correct prediction of step N slightly boosts step N+1, so the distribution advances naturally as actions repeat.

4. **Inventory delta suffix** (`| recent: spring entered`): gives the LLM temporal context unavailable from a static action string, disambiguating steps with identical gestures. Sticky accumulation prevents the delta from clearing during the 5–20 s inference latency.

5. **Multi-frame aggregation** for robot dispatch: a single HomePose frame can miss small objects (spring) due to momentary occlusion or confidence drops. Aggregating across 10–20 frames and requiring ≥2 appearances significantly reduces false negatives.

6. **Scene-consistency filter + augmented hint**: rather than simply discarding wrong predictions, the system builds a corrective prompt that names the incompatible objects and lists only the compatible next steps — the correction is injected into the very next inference call, not discarded silently.

7. **Displacement check in TriggerDetector**: a hand contact that produces no object movement is not a valid assembly action. The state machine ensures the learning phase only records steps that caused a measurable physical change (≥40 px displacement or object disappearance).
