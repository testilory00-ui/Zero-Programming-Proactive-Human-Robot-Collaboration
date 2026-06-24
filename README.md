# Proactive Robotic Assembly Assistant

A human–robot collaboration (HRC) system that lets an **ABB GoFa** collaborative arm
*proactively* fetch parts for a human operator performing a multi-step assembly. Two
Intel RealSense cameras watch the workspace; the system perceives the operator's
current action, predicts the next assembly step with a local/cloud **LLM** or a
**VLM**, and pre-stages the required parts from the robot's storage zone into the
operator's reach — before they are asked for.

The reference task is a **5-step carburetor assembly**. A new task can be taught from
a single demonstration (no manual annotation) via the learning phase.

> Master's thesis codebase. The two documents in [`docs/`](docs/) are the authoritative
> design reference: [`METHODOLOGY_REFERENCE.md`](docs/METHODOLOGY_REFERENCE.md) maps every
> subsystem to exact code, and [`THESIS_EXPERIMENTS_CONTEXT.md`](docs/THESIS_EXPERIMENTS_CONTEXT.md)
> documents the experimental design and metrics.

---

## How it works

The system runs as a multi-threaded producer–consumer pipeline. Three single-slot
queues decouple the 30 fps perception loop from slow (5–20 s) inference.

```
RealSense ×2 ──► PerceptionModule (MediaPipe hands + YOLOv8-seg)
                   │  semantic action string  ·  object tracking
                   ▼
            Dual-path routing
       ┌───────────────┴────────────────┐
   active assembly                   hands idle
       │                                 │
   LLM path                          VLM path
 Llama 4 Scout (HF API, default)     Gemini flash-lite (API)
   or Qwen3 4B INT8 (local OpenVINO/GPU, --local-model)
       └───────────────┬────────────────┘
                       ▼
           next-step prediction (+ optional StepTracker prior / scene filter)
                       ▼
        Robot worker ─► GraspContext (grasp-point routing) ─► TCP/RAPID ─► ABB GoFa
```

**Dual-path routing:** during active manipulation, frames are blurry and hands occlude
objects, so the LLM receives a compact *semantic action string*; when the operator's
hands are idle, the scene is read visually by the VLM from YOLO-annotated frames.

**Safety:** robot commands are dispatched only after a prediction passes a confirmation
counter; two suppression gates (last-step reached, workspace cleared) stop inference
when the assembly is complete.

Optional subsystems:
- **`--with-context`** — `StepTracker` maintains a softmax distribution over assembly
  steps (a Bayesian-style prior) and injects it into prompts to disambiguate steps.
- **`--with-filter`** — a scene-consistency filter validates each prediction against
  the objects YOLO actually sees and injects a corrective hint on mismatch.

---

## Repository layout

```
.
├── main.py                     # Inference orchestrator (assembly / disassembly)
├── perception.py               # PerceptionModule: hands + YOLO + semantic action
├── llm.py                      # LLM_planner: Qwen (local) / Llama (HF API) / Gemini (VLM)
├── step_tracker.py             # StepTracker: probabilistic step estimation (--with-context)
├── grasp_context.py            # Grasp-point registry & pixel↔robot mapping
├── robot_socket_client.py      # TCP/ASCII wire protocol to the RAPID controller
├── sheet_and_calibration.py    # ArUco homography calibration + GP sheet PDF
├── learn_procedure.py          # Learning phase (forward: observe assembly)
├── learn_procedure_inverse.py  # Learning phase (inverse: observe disassembly)
├── learn_utils.py              # Shared learning helpers
├── trial_logger.py             # Experiment trial logging
├── test_robot_socket.py        # Standalone socket test (no robot needed)
├── test_realsense.py           # Standalone camera test
├── learned_memory.json         # Active assembly task definition (loaded at runtime)
├── homography.json             # Saved camera→workspace homography
├── best_3.pt                   # YOLOv8-seg weights (assembly objects)
├── hand_landmarker.task        # MediaPipe hand-landmark model
├── sheet.pdf                   # Printable grasp-point calibration sheet
├── docs/                       # Methodology & experiment design references
├── logs/                       # Trial results, analysis scripts, figures
│   ├── analyze_results.py      #   evaluation pipeline → figures + metrics_summary.xlsx
│   ├── annotate_predictions.py #   post-hoc ground-truth annotation GUI
│   ├── match_videos.py         #   align trial videos for BORIS coding
│   └── analysis/               #   metrics xlsx, NASA-TLX, generated figures
├── RobotStudio/                # ABB RobotStudio station + RAPID modules
└── archive/                    # Legacy iterations & scratch (not part of the system)
```

---

## Requirements

- **Python 3.10+**
- 2× Intel RealSense cameras (perception + robot view) and an ABB GoFa controller for
  full operation. The analysis scripts and socket test run without any hardware.
- An **Intel GPU** (for the local OpenVINO model) is optional — the default LLM backend
  is the cloud HF Inference API.

```bash
pip install -r requirements.txt
```

### Environment variables

| Variable | Used for |
|----------|----------|
| `GEMINI_API_KEY` | VLM inference (Gemini) — **required** |
| `HUGGING_FACE_HUB_TOKEN` | Default LLM backend (Llama 4 Scout) — **required** |

> Both are validated at startup. The local model (`--local-model`) is **not** bundled
> in this repository (the ~4 GB OpenVINO export was removed to keep the repo lightweight).
> Re-export Qwen3 4B to `Qwen3_4B_INT8/` with Optimum-Intel to use that backend; otherwise
> run with the default HF API backend.

---

## Usage

All commands run from the repository root.

### Inference

```bash
python main.py                          # default backend (HF API / Llama 4 Scout)
python main.py --with-context           # inject StepTracker step prior into prompts
python main.py --with-filter            # scene-consistency filter on predictions
python main.py --with-context --with-filter
python main.py --local-model            # use local Qwen3 4B INT8 (OpenVINO/GPU)
python main.py --disassembly            # robot clears shared→remote zone (no LLM)
python main.py --silent-setup           # skip the grasp-point confirmation window
python main.py --force-recalibrate      # recompute the workspace homography
python main.py --perception-serial <SN> --robot-serial <SN>
```

Trial logging (writes `summary.json` + `predictions.jsonl` under `logs/`):

```bash
python main.py --with-context --participant-id P01 --trial-number 2
```

### Teach a new task

```bash
python learn_procedure.py               # observe an assembly, build learned_memory.json
python learn_procedure_inverse.py       # observe a disassembly, invert into assembly steps
python learn_procedure.py --webcam      # or use a webcam instead of RealSense
```

### Calibration

```bash
python sheet_and_calibration.py --generate    # generate the GP calibration sheet PDF
python sheet_and_calibration.py --calibrate    # ArUco homography calibration
python sheet_and_calibration.py --check        # verify calibration visually
```

### Experiment analysis

```bash
python logs/analyze_results.py --log-dir logs --output-dir output   # figures + xlsx
python logs/annotate_predictions.py                                  # ground-truth annotation
```

---

## Robot connection

`main.py` connects to the real robot at `192.168.125.1:1025` (hard-coded in `main()`).
For RobotStudio simulation, change the `RobotSocketClient(host=..., port=...)` arguments
to `127.0.0.1:5000`. The ABB project and RAPID modules are in [`RobotStudio/`](RobotStudio/);
RAPID binds a TCP server on port **1025** (port-forwarded to **5000** in the simulation
station).

---

## Notes

- The system is designed to be **run from the repository root** — model and config files
  (`best_3.pt`, `hand_landmarker.task`, `homography.json`, `learned_memory.json`) are
  resolved by relative path.
- `learned_memory.json` is the active task memory loaded at runtime.
- Trial **videos** and the **local model weights** are intentionally excluded from the
  repository (see `.gitignore`); the text trial results, analysis scripts, and figures
  are included.
