# Thesis Chapter: Experiments — Context and Reference Document

This document provides all the technical context needed to help write the "Experiments"
chapter of the thesis on a robotic assembly assistance system. It covers the experimental
design, every metric collected, the meaning of each generated figure, and all key numerical
results extracted from the trial logs.

**Important scope note**: This document focuses on system design, metrics, figures, and
numerical results. Code implementation details are intentionally omitted. The trial data
currently in the logs was collected by the thesis author across multiple sessions (not
independent participants); the IDs (P01, P02, …) reflect different recording days, not
different people. The actual participant study — including NASA-TLX subjective workload
evaluation — has not yet been run; those results will be added separately.

---

## 1. Experimental Design

### 1.1 Task

A 5-step carburetor assembly performed by a human operator assisted by an ABB GoFa
robotic arm. The robot fetches and delivers parts; the human performs the manipulation.

Assembly steps:
1. Place diaphragm
2. Insert spring
3. Attach cover
4. Adjust throttle stop
5. Install float bowl

### 1.2 Conditions

Five experimental conditions, each a distinct system configuration:

| Code | Label | Description |
|------|-------|-------------|
| `A_no_robot` | No Robot | Operator assembles alone — no robot assistance. Pure baseline; no inference calls are made. |
| `B_no_context` | No Context | Robot active. LLM/VLM inference runs but **no prior task context** is injected into the LLM prompt (the model does not know the assembly procedure). |
| `B_with_context` | With Context | Robot active. LLM prompt includes the **full task memory**: step descriptions, required objects, and a Bayesian prior probability over the current step derived from past confirmed actions. |
| `C_with_filter` | Filter | Robot active, no context. Adds a **scene-consistency filter**: after each LLM/VLM prediction the system checks whether the predicted next action is consistent with objects visible in the robot camera. Inconsistent predictions trigger a hint-injected retry. |
| `C_with_filter_and_context` | Filter + Context | Robot active. Combines the scene-consistency filter and the full task context (Bayesian prior). |

**Condition A has no logged trials in the current dataset** (collection pending).

### 1.3 Trial Distribution

| Condition | N trials |
|-----------|----------|
| B_no_context | 19 |
| B_with_context | 26 |
| C_with_filter | 4 |
| C_with_filter_and_context | 13 |
| **Total** | **62** |

These 62 trials were collected by the thesis author across multiple recording sessions
(not independent participants). The session IDs in the filenames (P01, P02, …) denote
different days/configurations, not different people. The formal participant study
(with independent subjects and NASA-TLX evaluation) is planned separately.

### 1.4 LLM Backends Used

Two LLM backends appear in the logs:
- `hf_api` → **Llama 4 Scout** (Hugging Face Inference API, cloud)
- `local` → **Qwen3 4B INT8** (local inference via OpenVINO on integrated GPU)

Both backends were tested across B and C conditions to compare cloud vs. local deployment.
VLM is always **Gemini** (cloud API).

---

## 2. Metrics

### 2.1 Trial-Level Metrics (one value per trial, stored in `*_summary.json`)

#### `atct_seconds` — Assembly Task Completion Time (ATCT)
- **Unit**: seconds
- **Definition**: wall-clock time from the trial start timestamp to the trial end timestamp. Covers the entire robot-assisted assembly, including all LLM/VLM inference calls, robot fetch trips, and human manipulation.
- **Interpretation**: lower is better (faster assembly with robot assistance). Condition A (no robot) serves as the human-alone baseline.
- **Note**: this is NOT a pure task time — it includes system latencies and robot movements.

#### `execution_success_score` — Execution Success Score
- **Unit**: float in [0, 1]
- **Definition**: fraction of the 5 assembly steps that were completed correctly and in the right order. Scored manually by the experimenter after each trial by reviewing video recordings. A value of 1.0 means all 5 steps were executed correctly; 0.6 means 3 out of 5 steps were correct.
- **Interpretation**: primary quality metric. Measures whether the robot helped the operator reach a complete and correct assembly, not just that the session ran without crashing.
- **Note**: the analysis code auto-normalizes integer inputs by dividing by N_STEPS (5) if raw counts are provided instead of fractions.

#### `vlm_escalation_rate` — VLM Escalation Rate
- **Unit**: fraction [0, 1]
- **Definition**: `vlm_call_count / total_call_count`. Measures what fraction of all inference calls had to escalate to the more expensive VLM (Gemini) rather than being resolved by the LLM alone.
- **Interpretation**: high escalation rate means the LLM frequently could not determine the assembly step on its own and required visual confirmation via the VLM. Lower is preferable (LLM handles more calls = faster, cheaper). However, some VLM calls are expected when there is genuine ambiguity in the scene.

#### `llm_call_count` / `vlm_call_count` / `total_call_count`
- **Unit**: integer count per trial
- **Definition**: number of LLM inference calls, VLM inference calls, and their total during the trial.
- **Interpretation**: reflects system activity. Fewer calls with high accuracy is ideal. More calls may indicate confusion (model repeatedly re-evaluating the same state).

#### `llm_mean_latency_s` / `llm_std_latency_s`
- **Unit**: seconds
- **Definition**: mean and standard deviation of wall-clock inference time across all LLM calls in the trial.
- **Interpretation**: reflects responsiveness. The Qwen3 local model runs in ~1–6 s on the local GPU; Llama 4 Scout (API) varies depending on network and API load. Both should stay well under the human action cycle (~2–5 s) to avoid blocking the operator.

#### `vlm_mean_latency_s` / `vlm_std_latency_s`
- **Unit**: seconds
- **Definition**: mean and standard deviation of Gemini VLM inference latency.
- **Interpretation**: VLM calls are generally slower than LLM calls (cloud round-trip + image processing). Latency spikes (e.g., one C_with_filter trial recorded mean=8.8 s, σ=8.2 s) indicate network variability.

### 2.2 Scene-Consistency Filter Metrics (only in C conditions)

These fields appear in `*_summary.json` only when `filter_enabled: true`.

#### `validation_failures_llm` / `validation_failures_vlm` / `validation_failures_total`
- **Definition**: number of times the scene-consistency filter rejected a prediction from LLM / VLM / either, during the trial. A "validation failure" means the predicted next operation required objects that were not detected in the robot camera frame at that moment.
- **Interpretation**: high counts indicate the filter is active and catching potentially wrong predictions. Combined with recovery rate (below), shows how effective the filter is.

#### `retries_triggered_llm` / `retries_triggered_vlm`
- **Definition**: number of retries triggered after a validation failure. Typically equals `validation_failures_*` (one retry per failure).

#### `hint_converged_valid_llm` / `hint_converged_valid_vlm`
- **Definition**: number of hint-injected retries where the model converged on a **valid** prediction (one consistent with the scene). This is the "recovery" event: the filter caught a wrong prediction, injected a hint about visible objects, and the model corrected itself.

#### `hint_converged_none_llm` / `hint_converged_none_vlm`
- **Definition**: number of hint-injected retries where the model responded with `"none"` (no step matched the visible scene). The filter accepted "none" as a safe fallback (no robot command dispatched).

#### `drops_after_hint_llm` / `drops_after_hint_vlm`
- **Definition**: number of hint-injected retries where the prediction was ultimately **dropped** (neither converged to valid nor to none — exceeded retry budget or failed to pass the filter). These calls produced no robot command.

### 2.3 Per-Call Metrics (stored in `*_predictions.jsonl`, one line per inference call)

#### `type` — Call Type
- Values: `"llm"` or `"vlm"`.
- Determines which model was invoked.

#### `inference_time_s` — Inference Latency
- **Unit**: seconds
- Wall-clock time for a single LLM or VLM inference call.

#### `semantic_action` — Semantic Action Input
- The discrete action label emitted by the perception module ("pinch", "assembly", "nothing") that was fed to the LLM as the primary signal.

#### `context_available` — Context Flag
- Boolean. True when the LLM was called with the full task memory and Bayesian prior (B_with_context, C_with_filter_and_context); False in no-context conditions.

#### `step_probabilities` — Bayesian Prior
- Dict: `{"step_1": float, "step_2": float, ..., "step_5": float}`.
- The probability distribution over the 5 assembly steps, computed from the confirmation history (how many times each step has been confirmed). This prior is fed into the LLM prompt to bias predictions toward the most likely next step. Only non-zero in "with_context" conditions.
- **Derived metrics**:
  - `max_prior_prob`: the probability mass on the most likely step (higher = more certain)
  - `prior_entropy` (bits): Shannon entropy of the distribution: `H = -Σ p log2(p)`
  - `normalized_entropy`: `H / log2(5)`, mapped to [0,1]. 0 = certain, 1 = maximum uncertainty (uniform distribution over 5 steps).
  - `prior_correct_prob`: the probability assigned to the ground-truth step — used to evaluate how well the tracker is calibrated.

#### `stage_of_assembly` — LLM/VLM Output
- The textual description of the current assembly stage returned by the model.
- Used to resolve the `predicted_step` (integer 1–5) via keyword overlap with `memory.json` step descriptions.

#### `next_operation` — Predicted Robot Action
- The specific robot command predicted by the model (e.g., "bring diaphragm", "none").
- If "none", no robot command is dispatched.

#### `ground_truth_step` — Ground Truth Annotation
- The manually annotated correct assembly step at the time of the call (integer 1–5, or "none").
- Assigned post-hoc by reviewing video recordings using the BORIS behavioral coding software.

#### `prediction_correct` — Correctness Flag
- Values: `True`, `False`, `"unclear"`.
- True if the model's `predicted_step` matches `ground_truth_step`.
- "unclear" when the ground truth could not be determined (e.g., transition between steps, ambiguous video frame).
- Used to compute `is_correct` (boolean, NaN for unclear).

#### `was_dispatched` — Robot Dispatch Flag
- Boolean. True if this call's prediction was ultimately sent to the robot (i.e., it passed the confirmation threshold of 2 consecutive identical predictions and, in C conditions, the scene-consistency filter).

#### `was_hint_injection` — Hint-Injected Call Flag
- Boolean. True if this call was a retry triggered after a filter validation failure (the model received an augmented prompt with a hint about visible objects).

#### `call_sequence_index` — Call Ordering
- Integer index within the trial (0 = first call). Used for x-axis in per-trial timeline plots.

### 2.4 Derived Trial-Level Metrics

#### `llm_accuracy` / `vlm_accuracy` / `overall_accuracy`
- Mean of `is_correct` across all annotated calls of each type within the trial.
- Measures how often the model correctly identified the current assembly step.

#### `unclear_rate`
- Fraction of calls annotated as "unclear" within the trial.

#### `was_dispatched` dispatch rate (per condition)
- `dispatched calls / total calls`. In B conditions = 1.0 (all calls are dispatched if they pass confirmation). In C conditions < 1.0 (some calls are dropped by the filter).

#### `not_assembly_ratio` (BORIS behavioral annotation)
- `non_assembly_time_s / atct_seconds`
- Computed from BORIS behavioral coding of video: annotator marks intervals where the operator is **not actively assembling** (waiting for robot, fetch trips, idle). This ratio measures how much of the task time the operator spent waiting or doing non-assembly activities. Lower is better (operator spends more time assembling, less time waiting for robot).

---

## 3. Figures — Description and Meaning

All figures are generated by `logs/analyze_results.py` and saved to `output/figures/`.

### fig01_overview.pdf
**Title**: "Task Completion Time and Execution Success per Condition"  
**Type**: Dual-axis bar+line chart  
**X-axis**: Condition (No Context, With Context, Filter, Filter+Context)  
**Left Y-axis**: ATCT in seconds (bars, colored by condition)  
**Right Y-axis**: Execution Success Score 0–100% (dashed line with diamond markers)  
**What it shows**: Primary outcome overview — simultaneously shows how long tasks took and how many steps were correctly executed. A condition that achieves lower ATCT AND higher success score is strictly better. Error bars on ATCT bars show ±1 SD across trials.

### fig01_accuracy_context.pdf
**Title**: "Prediction Accuracy by Condition"  
**Type**: Grouped bar chart  
**X-axis**: Condition (No Context, With Context, Filter, Filter+Context)  
**Y-axis**: Prediction accuracy (0–100%)  
**Groups per condition**: Llama 4 Scout LLM (API), Qwen3 4B INT8 LLM (local), VLM Gemini  
**What it shows**: How accurately each model (LLM/VLM, by backend) predicted the correct assembly step across conditions. Scatter dots show individual trial accuracies. Reveals whether context (Bayesian prior) improves LLM accuracy, and whether VLM is more or less accurate than LLM.

### fig02_accuracy_by_step.pdf
**Title**: "Prediction Accuracy per Assembly Step and Model Type"  
**Type**: Grouped bar chart  
**X-axis**: Assembly step (Step 1–5, with step description)  
**Y-axis**: Prediction accuracy (0–100%)  
**Groups**: LLM (red), VLM (purple)  
**What it shows**: Reveals which assembly steps are hardest to classify correctly. Some steps may be visually similar or have ambiguous semantic actions, causing lower accuracy. Bars with n<3 are shown faded (low statistical reliability).

### fig02_latency_distribution.pdf / fig03_latency_distribution.pdf
**Title**: "Inference Latency by Model"  
**Type**: Box-plot (with or without scatter dots — two versions)  
**X-axis**: Model category (Qwen3 4B INT8 local LLM, Llama 4 Scout API LLM, Gemini VLM)  
**Y-axis**: Inference time in seconds (log scale if range > 15×)  
**What it shows**: Latency distribution for each inference backend. Median is annotated. Shows whether models are fast enough for real-time HRC (reference: human action cycle ≈ 2–5 s). Compares local vs. cloud LLM trade-off.

### fig02_vlm_escalation.pdf
**Title**: "VLM Escalation Rate per Condition"  
**Type**: Bar + scatter (colored by execution success score via RdYlGn colormap)  
**X-axis**: B_no_context vs B_with_context  
**Y-axis**: VLM escalation rate (0–100%)  
**What it shows**: How often the LLM could not handle a call alone and escalated to the VLM. Each dot is a trial, colored by success score: green=success, red=failure. Allows checking if high escalation correlates with poor outcomes.

### fig03_atct_condition.pdf
**Title**: "Assembly Task Completion Time per Condition"  
**Type**: Strip plot with mean lines  
**X-axis**: All 5 conditions  
**Y-axis**: ATCT in seconds  
**What it shows**: Distribution of individual trial completion times per condition. Dashed horizontal line = condition mean. Individual points use different marker shapes per trial. Reveals spread and outliers.

### fig03_bayesian_heatmap.pdf / fig04_bayesian_evolution.pdf
**Title**: "Bayesian Belief Evolution — [participant] Trial [T##]"  
**Type**: Multi-panel line chart (3 panels stacked vertically):
1. **Top panel** (largest): 5 colored lines, one per assembly step, showing `step_probabilities["step_i"]` at each inference call index. Vertical dashed lines mark ground-truth step transitions (annotated "GT: Si").
2. **Middle strip**: Color-coded band showing the ground-truth step label at each call (colors match step lines).
3. **Bottom panel**: Correct (green triangle up) / wrong (red triangle down) markers at each call.  
**X-axis** (shared): Inference call index  
**Y-axis top**: P(current step = i), 0–100%  
**What it shows**: Demonstrates whether the Bayesian tracker correctly "walks" through the procedure. A well-behaved tracker shows the probability mass of step i rising as the operator enters step i and falling as they complete it. Step transitions should cause a smooth handover between adjacent step probabilities. Correctness markers show when the model's top prediction agreed with ground truth.

### fig04_confusion_matrix.pdf / fig05_confusion_matrix.pdf
**Title**: "Confusion Matrix: Predicted vs Ground Truth Step"  
**Type**: Row-normalized confusion matrix heatmap (one panel per model backend)  
- LLM panel: Blues colormap
- VLM panel: Purples colormap  
**Rows**: Ground truth step (S1–S5)  
**Columns**: Predicted step (S1–S5)  
**Cells**: raw count + row-normalized percentage  
**What it shows**: Which steps get confused with each other. The diagonal = correct predictions. Off-diagonal = confusions (e.g., step 3 predicted as step 2). Reveals systematic errors (e.g., model always predicts step 1 regardless of actual step = row i all pointing to column 1).

### fig05_dispatch_quality.pdf
**Title**: "Dispatch Quality: Accuracy vs Dispatch Rate"  
**Type**: Scatter plot  
**X-axis**: Dispatch rate = (calls that triggered a robot command) / (total calls)  
**Y-axis**: Accuracy of dispatched calls (fraction of dispatched predictions that were correct)  
**Points**: One per condition (large, opaque), plus individual trial dots (small, transparent)  
**What it shows**: The trade-off between selectivity and correctness. In B conditions dispatch rate=1.0 (all predictions go to robot). In C conditions the filter reduces dispatch rate but ideally increases accuracy (only passing predictions that the filter validates). Upper-right quadrant is ideal (dispatch everything and get it right). Upper-left is conservative-but-accurate. Lower-right is dangerous (dispatching wrong predictions).

### fig05_latency_accuracy_tradeoff.pdf / fig07_latency_accuracy_tradeoff.pdf
**Title**: "Latency–Accuracy Trade-off"  
**Type**: Scatter with error bars  
**X-axis**: Mean inference latency (s), ±1 SD  
**Y-axis**: Prediction accuracy, ±1 SD  
**Encoding**: Color = model (Qwen3 local = red, Llama 4 = amber, Gemini = purple); Marker shape = condition  
**What it shows**: Whether faster models sacrifice accuracy. Ideal position: low latency AND high accuracy (upper-left). Quantifies the engineering trade-off when choosing between local vs. cloud LLM.

### fig06_success_score.pdf
**Title**: "Task Execution Success Score per Condition"  
**Type**: Lollipop + strip chart  
**X-axis**: All 5 conditions  
**Y-axis**: Execution success score (0–100%)  
**What it shows**: Primary task quality metric per condition. Vertical line from 0 to condition mean, dot at mean, scatter dots for individual trials. Red dots flag failed trials (score < 50%). The "perfect" dashed reference at 100% shows how far each condition is from ideal.

### fig07_filter_effectiveness.pdf
**Title**: "Scene-Consistency Filter: Hint Injection Outcomes"  
**Type**: Two-panel figure  
**Panel A** (stacked horizontal bar): Mean filter activations per trial, broken into:
- Correct/recovered (green): hint injection led to a valid, correct prediction
- Wrong/not recovered (red): hint injection still produced wrong prediction
- Unclear/unannotated (grey): cannot determine outcome  
Individual trial total-activation dots shown below the bar.  
Title shows recovery rate percentage and mean±SD activations per trial.  
**Panel B** (vertical bar chart): Recovery rate per assembly step (for steps that triggered ≥1 filter activation). Bar height = fraction of filter-rejected calls at that step that were recovered. Step share shown as percentage (which step contributed what fraction of all filter activations).  
**What it shows**: How effective the scene-consistency filter is at correcting wrong predictions. A high recovery rate means the hint injection mechanism works. Per-step breakdown reveals which assembly steps cause the most filter activations.

### fig09_entropy_accuracy.pdf
**Title**: "Prior Uncertainty vs Prediction Correctness"  
**Type**: Scatter + logistic regression line  
**X-axis**: Normalized prior entropy (0 = certain about step, 1 = maximum uncertainty)  
**Y-axis**: Prediction correct (0 = wrong, 1 = correct) — binary  
**Points**: LLM (red) vs VLM (purple)  
**What it shows**: Tests whether the Bayesian tracker's confidence is calibrated. If the tracker works correctly, calls with low entropy (high certainty about which step we're in) should have higher prediction accuracy. A downward logistic curve (high entropy → low accuracy) validates this calibration.

### fig12_decision_timeline.pdf
**Title**: "Decision Timeline — Trial [T##]"  
**Type**: Gantt-style, 2-row time plot  
**Row 1** (GT step bands): Time axis divided into colored regions, one per assembly step, showing when each ground-truth step was active during the trial.  
**Row 2** (inference calls): Each LLM/VLM call shown as a vertical tick at its timestamp, colored by type (LLM=red, VLM=purple). Correctness shown via edge color (green=correct, red=wrong). A horizontal bar extends rightward by `inference_time_s` showing the latency span.  
**X-axis**: Elapsed time in seconds since trial start  
**What it shows**: A concrete single-trial trace that visualizes the interplay between human assembly progress (GT steps) and system inference calls over time. Reveals patterns like: calls clustering at step transitions, long latency calls that block the operator, VLM calls concentrated in certain steps.

### fig06_call_counts.pdf / fig10_call_counts.pdf
**Title**: "LLM / VLM Calls per Trial"  
**Type**: Grouped bar chart  
**X-axis**: Conditions  
**Y-axis**: Calls per trial  
**Groups**: LLM calls (red), VLM calls (purple), Total (grey)  
**What it shows**: System load per trial per condition. More calls = more compute. Reveals whether context or filter changes call volume. Error bars = ±1 SD.

---

## 4. Numerical Data Summary

All values below are aggregated from the 62 pre-study trials (author self-testing
across multiple sessions). These are system validation numbers, not results from an
independent participant study.

### 5.1 By Condition

#### B_no_context (N=19 trials)
| Metric | Value |
|--------|-------|
| ATCT | mean=186.5 s, σ=39.4 s, range=[110.2, 267.8] s |
| Execution success score | mean=0.92, range=[0.6, 1.0] |
| LLM calls/trial | mean=6.4 |
| VLM calls/trial | mean=6.3 |
| Total calls/trial | mean=12.6 |
| VLM escalation rate | mean=46.4% |
| LLM mean latency | mean=2.42 s |
| VLM mean latency | mean=3.40 s |
| LLM backends tested | Llama 4 Scout (API) and Qwen3 4B INT8 (local) |

Notable: widest ATCT range and lowest exec success — includes some failed trials
(exec=0.60; 3 out of 5 steps correct). One trial reached 24 total calls with 50% escalation.

#### B_with_context (N=26 trials)
| Metric | Value |
|--------|-------|
| ATCT | mean=190.8 s, σ=32.7 s, range=[136.4, 254.3] s |
| Execution success score | mean=0.98, range=[0.8, 1.0] |
| LLM calls/trial | mean=6.6 |
| VLM calls/trial | mean=6.6 |
| Total calls/trial | mean=13.2 |
| VLM escalation rate | mean=47.7% |
| LLM mean latency | mean=2.89 s |
| VLM mean latency | mean=2.45 s |

Notable: consistently near-perfect execution (mean 0.98 vs 0.92 without context).
Observed trials span ATCT from 136 s to 254 s; some trials achieved as few as 7 total
calls (4 LLM + 3 VLM) with escalation ~43%.

#### C_with_filter (N=4 trials)
| Metric | Value |
|--------|-------|
| ATCT | mean=201.3 s, σ=32.0 s, range=[158.8, 227.2] s |
| Execution success score | mean=1.00 (all trials perfect) |
| LLM calls/trial | mean=7.0 |
| VLM calls/trial | mean=5.0 |
| Total calls/trial | mean=12.0 |
| VLM escalation rate | mean=37.4% |
| LLM mean latency | mean=2.13 s |
| VLM mean latency | mean=4.07 s |

Filter stats (totals across 4 trials):
| Filter metric | LLM | VLM | Total |
|---------------|-----|-----|-------|
| Validation failures | 10 | 10 | 20 |
| Retries triggered | 10 | 10 | 20 |
| Hint recovered (valid) | 6 | 2 | 8 |
| Hint resolved to none | 0 | 0 | 0 |
| Dropped after hint | 6 | 5 | 11 |

Notable: one trial recorded VLM mean latency=8.8 s (σ=8.2 s) — severe network
variability spike. Small N (4 trials) means these filter numbers are illustrative only.

#### C_with_filter_and_context (N=13 trials)
| Metric | Value |
|--------|-------|
| ATCT | mean=169.5 s, σ=26.9 s, range=[115.7, 221.6] s |
| Execution success score | mean=1.00 (all trials perfect) |
| LLM calls/trial | mean=5.5 |
| VLM calls/trial | mean=4.2 |
| Total calls/trial | mean=9.6 |
| VLM escalation rate | mean=43.3% |
| LLM mean latency | mean=1.78 s |
| VLM mean latency | mean=2.74 s |

Filter stats (totals across 13 trials):
| Filter metric | LLM | VLM | Total |
|---------------|-----|-----|-------|
| Validation failures | 21 | 14 | 35 |
| Retries triggered | 21 | 14 | 35 |
| Hint recovered (valid) | 8 | 4 | 12 |
| Hint resolved to none | 0 | 0 | 0 |
| Dropped after hint | 8 | 6 | 14 |

Notable: one trial using Qwen3 local LLM recorded LLM latency=5.8 s vs ~1.3–2.9 s
for Llama 4 API across other trials — demonstrates the local vs. cloud latency gap.
One trial (Llama 4 API): ATCT=175.8 s, exec=1.00, 4 LLM + 4 VLM = 8 total calls,
filter failures=2.

### 5.2 Cross-Condition Highlights

- **Fastest condition**: C_with_filter_and_context (mean ATCT=169.5 s) — faster than B_with_context (190.8 s) and B_no_context (186.5 s), despite adding a filter check.
- **Best execution quality**: Both C conditions reach exec_success=1.00. B_no_context has the lowest at 0.92.
- **Fewest calls per trial**: C_with_filter_and_context (9.6/trial) vs 12–13 for B conditions — filter + context together reduce inference load.
- **Lowest VLM escalation**: C_with_filter (37.4%) — filter likely blocks some would-be VLM calls before they are counted.
- **Local vs. cloud LLM**: Qwen3 4B INT8 local ~5.8 s/call vs. Llama 4 Scout API ~1.3–2.9 s/call. Local is slower but fully offline.

### 5.3 Filter Recovery Rate Pooled (C conditions combined)
| Metric | C_with_filter | C_with_filter_and_context | Combined |
|--------|---------------|--------------------------|---------|
| Total validation failures | 20 | 35 | 55 |
| Total recoveries (hint → valid) | 8 | 12 | 20 |
| Recovery rate | 40% | 34% | ~36% |
| Dropped after hint | 11 | 14 | 25 |

---

## 5. Data Files

| File pattern | Content |
|---|---|
| `logs/*_summary.json` | One JSON per trial — all trial-level metrics |
| `logs/*_predictions.jsonl` | One JSONL per trial — one record per LLM/VLM call |
| `logs/*_frames/*.jpg` | Camera frames captured at each VLM call |
| `logs/video/*.MOV` | Trial recordings for BORIS behavioral annotation |
| `logs/analyze_results.py` | Full analysis pipeline: DataLoader → MetricsEngine → FigureFactory |
| `logs/annotate_predictions.py` | Post-hoc tool to assign `ground_truth_step` by reviewing frames |
| `logs/match_videos.py` | Renames video files to match trial identifiers for BORIS |
| `Metrics.xlsx` | Exported metrics spreadsheet (project root) |
| `memory.json` / `learned_memory.json` | 5-step carburetor assembly task definition used in LLM prompts |

---

## 6. Naming Conventions

Trial filenames follow the pattern:
```
trial_P<session>_T<NN>_<condition>_summary.json
```
Where:
- `P<session>` = recording session ID (e.g., P01, P02 — all from the same author, different days)
- `T<NN>` = trial number within that session
- `<condition>` = one of `A_no_robot`, `B_no_context`, `B_with_context`, `C_with_filter`, `C_with_filter_and_context`

---

## 7. Key Interpretation Points for the Thesis Chapter

1. **Context improves reliability**: B_with_context exec_success=0.98 vs B_no_context=0.92. The Bayesian prior helps the LLM stay on track.

2. **Filter achieves perfect execution**: Both C conditions reach exec_success=1.00. The scene-consistency filter eliminates the failed trials seen in B_no_context.

3. **Filter + context is fastest**: Despite adding a filter check, C_with_filter_and_context achieves the lowest mean ATCT (169.5 s) because it dispatches fewer, more confident robot commands (9.6 calls/trial vs 12–13 for B conditions).

4. **VLM escalation is a necessary cost**: ~40–48% of calls reach the VLM in B conditions. This is expected — 10-frame windows that contain ambiguous assembly states genuinely need visual grounding.

5. **Filter recovery rate ~36%**: When the filter activates and injects a hint, the model recovers correctly about 1 in 3 times. The other activations either drop the command or produce "none" — both safe outcomes (no wrong robot action).

6. **Latency is sub-human-cycle for API LLM**: Llama 4 Scout (API) shows mean ~1–2 s per call (across most trials), well within the human action cycle. Qwen3 local is ~2–6 s. Both allow the system to react before the operator finishes the next action.

7. **Bayesian entropy calibration**: The entropy vs. correctness analysis (fig09) tests whether the tracker's confidence correlates with actual correctness — a key validity check for the prior mechanism.

8. **ATCT does not include robot fetch time** explicitly (noted as pending in the code). The ATCT is total wall-clock time, which implicitly includes robot fetch trips.

9. **Condition A baseline pending**: The baseline (human alone, no robot) was not yet collected at analysis time. The thesis should note this limitation.

10. **Small sample for C_with_filter** (N=4): Results should be interpreted cautiously — filter effectiveness statistics have high variance with only 4 trials.
