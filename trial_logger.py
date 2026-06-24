"""
trial_logger.py
Experimental trial logger for HRC thesis evaluation.

Logs per-call LLM/VLM predictions and trial-level summary metrics
to support post-hoc analysis of:
  - ATCT (Assembly Task Completion Time)
  - LLM/VLM call counts and inference latencies
  - LLM prediction accuracy (with vs without StepTracker context)
  - VLM escalation rate
  - Ground truth annotation (ground_truth_step field, filled post-hoc)

─── OUTPUT FILES ────────────────────────────────────────────────────────────

Three artefacts are written per run (tag = trial_<ID>_T<N>_<condition>):

  logs/<tag>_summary.json          — one JSON object with trial-level metrics:
    {
      "trial_id":               "P001",
      "trial_number":           1,
      "condition":              "B_with_context",
      "timestamp_start":        "<ISO-8601>",
      "timestamp_end":          "<ISO-8601>",
      "atct_seconds":           120.5,          // Assembly Task Completion Time
      "llm_backend":            hf_api
      "vlm_backend":            gemini
      "llm_call_count":         8,
      "vlm_call_count":         3,
      "total_call_count":       11,
      "vlm_escalation_rate":    0.2727,          // vlm_count / total_count
      "llm_mean_latency_s":     2.1,
      "llm_std_latency_s":      0.3,
      "vlm_mean_latency_s":     4.5,
      "vlm_std_latency_s":      0.8,
      "execution_success_score": 0.6,            // correct_steps/total_steps, entered by operator at trial end
      "notes":                  ""
    }

  logs/<tag>_predictions.jsonl     — one JSON record per line, one per LLM/VLM call:
    {
      "call_id":           1,
      "type":              "llm" | "vlm",
      "timestamp":         "<ISO-8601>",
      "inference_time_s":  1.234,
      "input": {
        "semantic_action":    "<action label>",   // "non-assembly / scene ambiguous" for VLM
        "context_available":  true | false,       // was StepTracker context injected?
        "step_probabilities": {"step_1": 0.72, "step_2": 0.12, ...},  // {} if no context
        "frame_saved":        "<path>"            // VLM only — composite JPEG path
      },
      "output": {
        "stage_of_assembly": "<current step description>",
        "next_operation":    "<next step description>",
        "objects_required":  ["obj_a", "obj_b"]
      },
      "ground_truth_step":  null,                 // filled post-hoc
      "prediction_correct": null                  // filled post-hoc
    }

  logs/<tag>_frames/               — composite JPEGs (up to 5 frames side-by-side,
                                     1280×256 px), one per VLM call, named
                                     call_<NNN>_vlm.jpg. Used for post-hoc
                                     ground truth annotation.
"""

import json
import os
import time
import datetime
import statistics

import cv2
import numpy as np


class TrialLogger:
    def __init__(self, participant_id: str, condition: str,
                 trial_number: int, log_dir: str = "logs",
                 llm_backend: str = "hf_api",
                 filter_enabled: bool = False):
        """
        Parameters
        ----------
        participant_id : str
            Participant identifier, e.g. "P001".
        condition : str
            One of: "A_no_robot" | "B_no_context" | "B_with_context" |
            "C_with_filter" | "C_with_filter_and_context".
        trial_number : int
            Sequential trial index for this participant (1-based).
        log_dir : str
            Root directory for all log files. Created if absent.
        llm_backend : str
            LLM inference backend used for this run.
            "local" = local OpenVINO model (Qwen INT4/INT8),
            "hf_api" = Hugging Face Inference API (default).
            VLM is always Gemini API.
        """
        valid_conditions = {"A_no_robot", "B_no_context", "B_with_context",
                            "C_with_filter", "C_with_filter_and_context"}
        if condition not in valid_conditions:
            raise ValueError(
                f"[TrialLogger] Invalid condition '{condition}'. "
                f"Must be one of: {sorted(valid_conditions)}"
            )

        self.participant_id = participant_id
        self.condition      = condition
        self.trial_number   = trial_number
        self.llm_backend    = llm_backend
        self.filter_enabled = filter_enabled

        tag = f"trial_{participant_id}_T{trial_number:02d}_{condition}"
        self.log_dir           = log_dir
        self.summary_path      = os.path.join(log_dir, f"{tag}_summary.json")
        self.predictions_path  = os.path.join(log_dir, f"{tag}_predictions.jsonl")
        self.frames_dir        = os.path.join(log_dir, f"{tag}_frames")

        os.makedirs(log_dir,         exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

        self._call_counter   = 0
        self._start_time     = None
        self._end_time       = None
        self._predictions_fh = open(self.predictions_path, "w", encoding="utf-8")

        self.llm_times: list[float] = []
        self.vlm_times: list[float] = []
        self.llm_count = 0
        self.vlm_count = 0

        # Scene-consistency filter counters
        self.validation_failures_llm   = 0
        self.validation_failures_vlm   = 0
        self.retries_triggered_llm     = 0
        self.retries_triggered_vlm     = 0
        self.hint_converged_valid_llm  = 0
        self.hint_converged_valid_vlm  = 0
        self.hint_converged_none_llm   = 0
        self.hint_converged_none_vlm   = 0
        self.drops_after_hint_llm      = 0
        self.drops_after_hint_vlm      = 0

        print(f"[TrialLogger] Initialised — {tag}")
        print(f"[TrialLogger] Summary     → {self.summary_path}")
        print(f"[TrialLogger] Predictions → {self.predictions_path}")

    # ------------------------------------------------------------------
    # Trial timing
    # ------------------------------------------------------------------

    def start_trial(self):
        """
        Call when the experimenter signals the start of the trial (e.g. presses Enter).
        Records wall-clock start time.
        """
        self._start_time = time.time()
        print(f"[TrialLogger] *** TRIAL STARTED — {self.participant_id} / "
              f"{self.condition} / T{self.trial_number:02d} ***")

    def stop_trial(self):
        """
        Call when the participant says 'done' (or experimenter stops the trial).
        Records wall-clock end time and prints ATCT.
        """
        self._end_time = time.time()
        print(f"[TrialLogger] *** TRIAL STOPPED — ATCT = {self.atct:.1f}s ***")

    @property
    def atct(self) -> float:
        """Assembly Task Completion Time in seconds. Returns 0 if not yet started/stopped."""
        if self._start_time is None or self._end_time is None:
            return 0.0
        return round(self._end_time - self._start_time, 2)

    @property
    def trial_running(self) -> bool:
        """True between start_trial() and stop_trial()."""
        return self._start_time is not None and self._end_time is None

    # ------------------------------------------------------------------
    # Prediction logging
    # ------------------------------------------------------------------

    def log_llm_call(self,
                     semantic_action: str,
                     step_probabilities: dict,
                     llm_output: dict,
                     inference_time: float,
                     context_available: bool,
                     was_hint_injection: bool = False):
        """
        Log one LLM (Qwen) inference call.

        Parameters
        ----------
        semantic_action : str
            The dominant hand action string from the perception buffer
            (e.g. "assembly pinch on diaphragm").
        step_probabilities : dict
            StepTracker softmax probabilities at call time,
            e.g. {"step_1": 0.05, "step_2": 0.72, ...}.
            Pass empty dict {} when context is not available.
        llm_output : dict
            Parsed JSON output from the LLM, containing keys:
            "stage of assembly", "next operation", "objects required".
        inference_time : float
            Wall-clock seconds from call start to result received.
        context_available : bool
            True if StepTracker context was injected into the prompt.
        """
        self._call_counter += 1
        self.llm_count     += 1
        self.llm_times.append(inference_time)

        record = {
            "call_id":          self._call_counter,
            "type":             "llm",
            "timestamp":        datetime.datetime.now().isoformat(),
            "inference_time_s": round(inference_time, 3),
            "input": {
                "semantic_action":    semantic_action,
                "context_available":  context_available,
                "step_probabilities": step_probabilities,
            },
            "output": {
                "stage_of_assembly": llm_output.get("stage of assembly",
                                      llm_output.get("stage_of_assembly", "")),
                "next_operation":    llm_output.get("next operation",
                                      llm_output.get("next_operation", "")),
                "objects_required":  llm_output.get("objects required",
                                      llm_output.get("objects_required", [])),
            },
            # Filled post-hoc during video annotation:
            "ground_truth_step":   None,
            "prediction_correct":  None,
            "was_hint_injection":  was_hint_injection,
        }

        self._write_record(record)

    def log_vlm_call(self,
                     frames: list,
                     step_probabilities: dict,
                     vlm_output: dict,
                     inference_time: float,
                     context_available: bool,
                     was_hint_injection: bool = False):
        """
        Log one VLM (Gemini) inference call.
        Saves a composite JPEG of the input frames for post-hoc ground truth annotation.

        Parameters
        ----------
        frames : list of np.ndarray
            The list of BGR frames sent to Gemini (up to 5).
        step_probabilities : dict
            StepTracker softmax probabilities at call time.
            Pass empty dict {} when context is not available.
        vlm_output : dict
            Parsed JSON output from Gemini.
        inference_time : float
            Wall-clock seconds from call start to result received.
        context_available : bool
            True if StepTracker context was injected into the prompt.
        """
        self._call_counter += 1
        self.vlm_count     += 1
        self.vlm_times.append(inference_time)

        frame_filename = f"call_{self._call_counter:03d}_vlm.jpg"
        frame_path     = os.path.join(self.frames_dir, frame_filename)
        self._save_composite_frame(frames, frame_path)

        record = {
            "call_id":          self._call_counter,
            "type":             "vlm",
            "timestamp":        datetime.datetime.now().isoformat(),
            "inference_time_s": round(inference_time, 3),
            "input": {
                "semantic_action":    "non-assembly / scene ambiguous",
                "context_available":  context_available,
                "step_probabilities": step_probabilities,
                "frame_saved":        frame_path,
            },
            "output": {
                "stage_of_assembly": vlm_output.get("stage of assembly",
                                      vlm_output.get("stage_of_assembly", "")),
                "next_operation":    vlm_output.get("next operation",
                                      vlm_output.get("next_operation", "")),
                "objects_required":  vlm_output.get("objects required",
                                      vlm_output.get("objects_required", [])),
            },
            # Filled post-hoc during video annotation:
            "ground_truth_step":  None,
            "prediction_correct": None,
            "was_hint_injection": was_hint_injection,
        }

        self._write_record(record)

    # ------------------------------------------------------------------
    # Filter event logging
    # ------------------------------------------------------------------

    def log_validation_failure(self,
                                call_type: str,
                                predicted_plan: dict,
                                missing_objects: list,
                                confirmed_classes: set,
                                context_str: str,
                                was_hint_injection: bool,
                                augmented_hint_built: str = ""):
        """Log one scene-consistency filter rejection to the predictions JSONL."""
        if call_type == "llm":
            self.validation_failures_llm += 1
            self.retries_triggered_llm   += 1
        else:
            self.validation_failures_vlm += 1
            self.retries_triggered_vlm   += 1

        import re as _re
        step_probs = {}
        if context_str:
            for m in _re.finditer(r"Step (\d+):\s+(\d+)%\s+chance current", context_str):
                step_probs[f"step_{m.group(1)}"] = int(m.group(2)) / 100.0

        record = {
            "event_type":                   "validation_failure",
            "timestamp":                    datetime.datetime.now().isoformat(),
            "call_type":                    call_type,
            "was_hint_injection":           was_hint_injection,
            "predicted_next_operation":     (predicted_plan.get("next_operation")
                                             or predicted_plan.get("next operation", "")),
            "predicted_stage_of_assembly":  (predicted_plan.get("stage_of_assembly")
                                             or predicted_plan.get("stage of assembly", "")),
            "predicted_objects_required":   (predicted_plan.get("objects_required")
                                             or predicted_plan.get("objects required", [])),
            "missing_objects":              list(missing_objects),
            "confirmed_classes_in_scene":   sorted(confirmed_classes),
            "step_tracker_probabilities":   step_probs,
            "augmented_hint_built":         augmented_hint_built,
        }
        self._write_record(record)

    def log_dispatch(self, dispatched_from_call_id: int, next_operation: str):
        """Log a robot dispatch event so post-hoc analysis can compute dispatch accuracy."""
        record = {
            "event_type":               "dispatch",
            "timestamp":                datetime.datetime.now().isoformat(),
            "dispatched_from_call_id":  dispatched_from_call_id,
            "next_operation":           next_operation,
        }
        self._write_record(record)

    def log_hint_outcome(self, call_type: str, outcome: str):
        """Update convergence counters after a hint-injected call is processed.

        Parameters
        ----------
        call_type : str  — "llm" or "vlm"
        outcome   : str  — "valid" | "none" | "invalid_dropped"
        """
        if call_type == "llm":
            if outcome == "valid":
                self.hint_converged_valid_llm += 1
            elif outcome == "none":
                self.hint_converged_none_llm += 1
            else:
                self.drops_after_hint_llm += 1
        else:
            if outcome == "valid":
                self.hint_converged_valid_vlm += 1
            elif outcome == "none":
                self.hint_converged_none_vlm += 1
            else:
                self.drops_after_hint_vlm += 1

    # ------------------------------------------------------------------
    # Summary export
    # ------------------------------------------------------------------

    def save_summary(self, execution_score: int | None = None):
        """
        Write the trial summary JSON file.
        Call this in the finally block of main(), after the assembly loop exits.

        Parameters
        ----------
        execution_score : float | None
            Correct steps / total steps, computed from the operator's input at
            trial end (e.g. 3/5 → 0.6). None if total steps could not be determined.
        """
        total_calls = self.llm_count + self.vlm_count

        summary = {
            "trial_id":       self.participant_id,
            "trial_number":   self.trial_number,
            "condition":      self.condition,

            "timestamp_start": (datetime.datetime.fromtimestamp(self._start_time).isoformat()
                                if self._start_time else None),
            "timestamp_end":   (datetime.datetime.fromtimestamp(self._end_time).isoformat()
                                if self._end_time else None),
            "atct_seconds":    self.atct,

            "llm_backend":         self.llm_backend,
            "vlm_backend":         "gemini_api",

            "llm_call_count":      self.llm_count,
            "vlm_call_count":      self.vlm_count,
            "total_call_count":    total_calls,
            "vlm_escalation_rate": (round(self.vlm_count / total_calls, 4)
                                    if total_calls > 0 else None),

            "llm_mean_latency_s": (round(statistics.mean(self.llm_times), 3)
                                   if self.llm_times else None),
            "llm_std_latency_s":  (round(statistics.stdev(self.llm_times), 3)
                                   if len(self.llm_times) > 1 else None),
            "vlm_mean_latency_s": (round(statistics.mean(self.vlm_times), 3)
                                   if self.vlm_times else None),
            "vlm_std_latency_s":  (round(statistics.stdev(self.vlm_times), 3)
                                   if len(self.vlm_times) > 1 else None),

            # Entered by the operator at trial end:
            "execution_success_score": execution_score,
            "notes": "",                        # free text for anomalies

            # Scene-consistency filter stats (all zero when filter_enabled=False)
            "filter_enabled":              self.filter_enabled,
            "validation_failures_llm":     self.validation_failures_llm,
            "validation_failures_vlm":     self.validation_failures_vlm,
            "validation_failures_total":   self.validation_failures_llm + self.validation_failures_vlm,
            "retries_triggered_llm":       self.retries_triggered_llm,
            "retries_triggered_vlm":       self.retries_triggered_vlm,
            "hint_converged_valid_llm":    self.hint_converged_valid_llm,
            "hint_converged_valid_vlm":    self.hint_converged_valid_vlm,
            "hint_converged_none_llm":     self.hint_converged_none_llm,
            "hint_converged_none_vlm":     self.hint_converged_none_vlm,
            "drops_after_hint_llm":        self.drops_after_hint_llm,
            "drops_after_hint_vlm":        self.drops_after_hint_vlm,
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[TrialLogger] Summary saved    → {self.summary_path}")
        print(f"[TrialLogger] Predictions saved → {self.predictions_path}")
        print(f"[TrialLogger] ATCT={self.atct:.1f}s  "
              f"LLM={self.llm_count}  VLM={self.vlm_count}")

    def close(self):
        """Close the open predictions file handle. Always call this in the finally block."""
        if not self._predictions_fh.closed:
            self._predictions_fh.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_record(self, record: dict):
        """Append one JSON record to the JSONL predictions file and flush immediately."""
        self._predictions_fh.write(json.dumps(record) + "\n")
        self._predictions_fh.flush()

    @staticmethod
    def _save_composite_frame(frames: list, path: str,
                               target_w: int = 1280, target_h: int = 256):
        """
        Save up to 5 frames side-by-side as a single JPEG.
        Used for VLM ground truth annotation during post-hoc analysis.
        Each frame is resized to (target_w // n_frames, target_h).
        """
        if not frames:
            return
        n    = min(len(frames), 5)
        w    = target_w // n
        h    = target_h
        imgs = []
        for i in range(n):
            resized = cv2.resize(frames[i], (w, h))
            imgs.append(resized)
        composite = np.hstack(imgs)
        cv2.imwrite(path, composite)
