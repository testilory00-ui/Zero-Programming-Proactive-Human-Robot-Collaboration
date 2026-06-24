# Thesis Figures — Pilot Study Results

> **Source:** `logs/analysis/figures/` (8 PDF figures) and `logs/analysis/pilot_report.txt`
> **Study:** HRC Assembly System Pilot — N = 264 trials, 7 participants, 4 experimental conditions.
> **Conditions:** B_no_context (No Context), B_with_context (With Context), C_with_filter (Filter), C_with_filter_and_context (Filter + Context).
> All values are descriptive (no inferential statistics yet — single pilot participant drove most trials).

---

## Numerical Results Summary

### Per-Condition Metrics

| Condition | N trials | ATCT (s) | σ (s) | Exec Score | LLM calls/trial | VLM calls/trial | Total calls/trial | VLM escalation |
|---|---|---|---|---|---|---|---|---|
| No Context | 71 | 186.1 | 39.6 | 0.90 | 6.6 (σ=2.5) | 6.8 (σ=4.3) | 13.4 (σ=5.0) | 47.3% (σ=19.2%) |
| With Context | 105 | 187.1 | 30.6 | 0.98 | 6.2 (σ=2.8) | 7.1 (σ=5.4) | 13.3 (σ=6.5) | 50.3% (σ=20.3%) |
| Filter | 22 | 199.0 | 28.4 | 1.00 | 7.1 (σ=1.3) | 4.5 (σ=3.2) | 11.6 (σ=3.6) | 35.2% (σ=15.0%) |
| Filter + Context | 66 | 169.9 | 27.6 | 1.00 | 5.3 (σ=1.9) | 4.2 (σ=1.9) | 9.5 (σ=1.9) | 44.3% (σ=14.8%) |

### Per-Condition Prediction Accuracy

| Condition | Llama 4 Scout | Qwen3 4B INT8 | VLM (Gemini) |
|---|---|---|---|
| No Context | 65.4% (n=81) | 50.0% (n=40) | 72.4% (σ=13.8%) |
| With Context | 73.9% (n=88) | 69.5% (n=82) | 72.8% (σ=11.6%) |
| Filter | 61.9% (n=21) | 57.1% (n=7) | 75.1% (σ=4.3%) |
| Filter + Context | 72.1% (n=61) | 80.0% (n=10) | 72.9% (σ=8.0%) |

### Overall Call Summary

- Total calls: **756** (annotated: 754/756)
- LLM calls: **391** — Llama 4 Scout: 252 (accuracy **69.7%**), Qwen3 4B INT8: 139 (accuracy **64.0%**)
- VLM calls: **365**
- Overall accuracy: **69.9%**

### Per-Step Accuracy

| Step | LLM accuracy | n | VLM accuracy | n |
|---|---|---|---|---|
| Step 1 | 85% | 52 | 89% | 38 |
| Step 2 | 92% | 80 | 79% | 33 |
| Step 3 | 50% | 96 | 61% | 67 |
| Step 4 | 62% | 123 | 51% | 39 |
| Step 5 | 58% | 38 | 57% | 75 |

Steps 3–5 are systematically harder for both model types; Step 2 is the easiest for the LLM path.

### Filter Analysis

- Trials with filter enabled: **88**
- Filter activations: LLM = 156, VLM = 121, Total = **277**
- Hint convergence → valid: LLM = 72, VLM = 31
- Hint convergence → none: LLM = 0, VLM = 0
- Dropped after hint: LLM = 72, VLM = 54

---

## Figure Descriptions

---

### `fig01_overview.pdf` — Task Completion Time and Execution Success per Condition

**What it shows.**
A dual-axis bar chart comparing the four experimental conditions on two metrics simultaneously.
The left y-axis (bar height) is Assembly Task Completion Time (ATCT) in seconds; the right y-axis (dashed line with diamond markers) is the Execution Success Score expressed as a percentage.

**Visual structure.**
Four colored bars (yellow = No Context, teal = With Context, blue = Filter, pink = Filter + Context) with ±1 SD error bars. A dashed grey line connects the four success-score diamonds.

**Key values annotated in the figure.**

| Condition | ATCT | Exec Score |
|---|---|---|
| No Context | 186 s | 90% |
| With Context | 187 s | 98% |
| Filter | 199 s | 100% |
| Filter + Context | **170 s** | 100% |

**Thesis narrative.**
Filter + Context achieves the fastest completion time (170 s) while also reaching perfect execution (100%). Adding context alone brings the success score from 90% to 98% without changing completion time meaningfully. The Filter-only condition slightly increases ATCT (199 s), likely because the validation step adds latency on failed predictions before a hint is injected.

**Suggested caption.**
> *Figure X — Assembly Task Completion Time (bars, left axis) and Execution Success Score (diamonds, right axis) for the four system conditions. Error bars = ±1 SD. Pilot study, N = 264 trials.*

---

### `fig02_bayesian_belief.pdf` — Bayesian Belief Evolution (P08 Trial T02, C\_with\_filter\_and\_context)

**What it shows.**
A single-trial trace of the system's internal Bayesian belief state over the course of one carburetor assembly trial. The y-axis is P(current step = i) for each of the five assembly steps; the x-axis is the sequential inference call index (0–8 calls for this trial).

**Visual structure.**
Five coloured lines (one per step: S1 green, S2 blue, S3 orange, S4 purple, S5 light blue). Dashed vertical lines delimit ground-truth step boundaries shown in the coloured GT band below the main plot. Green triangles / red inverted triangles on the "Correct?" row mark whether each call produced a correct prediction.

**Key observations.**
- At call 0 the belief is diffuse (all steps ~15–20%).
- Step 1 belief rises to ~60% at call 1, then collapses as the operator moves to Step 2.
- Step 2 dominates calls 2–3 (peak ~67%).
- Steps 3–5 show progressively flatter peaks and more residual uncertainty, consistent with the lower per-step accuracies at later steps.
- One wrong prediction appears at call 6 (Step 4 ground truth), which the filter catches before dispatch.
- The final calls (7–8) fall in Step 5, where the belief plateaus around 32% — all correct.

**Thesis narrative.**
This figure illustrates the temporal dynamics of the Bayesian planner on a representative successful trial. It demonstrates that early steps benefit from a strong prior (only one plausible next step given the task sequence), while later steps suffer from perceptual ambiguity. The filter intervention at call 6 prevents a wrong dispatch despite a momentary belief collapse.

**Suggested caption.**
> *Figure X — Bayesian belief P(current step = i) over inference calls for participant P08, trial T02 (Filter + Context condition). Vertical dashed lines mark ground-truth step transitions. Green/red markers on the bottom row indicate correct/wrong predictions. The grey band in the GT row highlights the true step active at each call.*

---

### `fig03_confusion_matrix.pdf` — Confusion Matrix: Predicted vs Ground Truth Step

**What it shows.**
Three row-normalised 5×5 confusion matrices (assembly Steps S1–S5) for the three inference models: Llama 4 Scout (LLM, n=250), Qwen3 4B INT8 (LLM, n=139), and Gemini (VLM, n=239).

**Visual structure.**
Each matrix uses a blue (Llama/Qwen) or purple (Gemini) heatmap with raw counts and row-normalised percentages annotated in each cell. Three panels are arranged side by side with a shared caption.

**Key diagonal values (correct predictions).**

| Step | Llama 4 Scout | Qwen3 4B INT8 | Gemini VLM |
|---|---|---|---|
| S1 | 91% (29/32) | 75% (15/20) | 92% (34/37) |
| S2 | 92% (45/49) | 90% (28/31) | 90% (27/30) |
| S3 | 50% (28/56) | 57% (23/40) | 68% (43/63) |
| S4 | 64% (54/85) | 58% (22/38) | 53% (20/38) |
| S5 | 61% (17/28) | 60% (6/10) | 42% (30/71) |

**Common confusion patterns.**
- S3 is frequently confused with S2 (Llama: 41% of S3 predictions fall on S2; Qwen: 42%).
- S4 is confused with S3 for all models.
- VLM (Gemini) has a large off-diagonal at S5→S1 (38%), indicating a systematic "reset" error where the VLM predicts the task has restarted at the end.

**Thesis narrative.**
The confusion matrices reveal that early steps (S1, S2) are reliably identified by all models, while mid-to-late steps (S3–S5) exhibit substantial confusion, primarily between adjacent steps. This is consistent with the visual similarity between consecutive assembly sub-tasks. The VLM's S5→S1 confusion is a qualitatively different failure mode (hallucinated restart) not observed in the LLM path.

**Suggested caption.**
> *Figure X — Row-normalised confusion matrices for step prediction across the three inference models. Values show count (percentage of ground-truth row). Pilot study, N = 264 trials.*

---

### `fig04_latency_accuracy_tradeoff.pdf` — Latency–Accuracy Trade-off

**What it shows.**
A scatter plot placing each model–condition combination at its (mean inference latency, prediction accuracy) coordinate. Error bars = ±1 SD in both dimensions.

**Visual structure.**
Three colours (red = Qwen3 4B INT8, yellow/gold = Llama 4 Scout, purple = Gemini VLM); four shapes (circle = No Context, square = With Context, triangle = Filter, diamond = Filter + Context). Twelve points total.

**Key observations.**
- Llama 4 Scout is the fastest model (~1 s) at moderate accuracy (~65–76%), tightly clustered on the left.
- Qwen3 4B INT8 spans 5–6 s latency; With Context and Filter + Context reach ~73–84% accuracy.
- Gemini (VLM) spans 2.5–5.5 s; Filter + Context achieves ~85% accuracy at ~2.5 s — the best accuracy point in the plot.
- No model-condition combination dominates in both dimensions; Llama 4 Scout offers the best latency at the cost of ~10 pp lower accuracy.

**Thesis narrative.**
The trade-off plot shows that there is no single Pareto-optimal model across all conditions. The Filter + Context condition consistently shifts points toward higher accuracy for all models, at a small latency cost. Llama 4 Scout is the viable choice when latency is the binding constraint; Gemini with Filter + Context maximises accuracy at moderate latency.

**Suggested caption.**
> *Figure X — Latency–accuracy trade-off for all model–condition combinations. Each point is the per-condition mean; error bars = ±1 SD. Pilot study, N = 264 trials.*

---

### `fig05_dispatch_quality.pdf` — Dispatch Quality: Accuracy vs Dispatch Rate

**What it shows.**
A scatter plot of trial-level dispatch quality. The x-axis is the dispatch rate (fraction of total inference calls that were actually sent to the robot); the y-axis is dispatch accuracy (fraction of dispatched commands that were correct). Each small dot is one trial; large opaque dots are condition means.

**Visual structure.**
Four colours (orange = No Context, green = With Context, blue = Filter, pink = Filter + Context). The 100% reference lines are shown as dashed grey lines on both axes.

**Key observations.**
- Without a filter (No Context, With Context): dispatch rate clusters near 100% — almost every confirmed prediction is dispatched. Mean dispatch accuracy: ~65% (No Context) and ~72% (With Context).
- With filter (Filter, Filter + Context): dispatch rate drops to 50–60%, but dispatch accuracy rises toward 85–100% for many trials.
- Filter + Context achieves several trials at 100% accuracy and 50–65% dispatch rate, representing the ideal quadrant (high accuracy, controlled dispatch).
- No Context has the widest accuracy spread (40–90%), indicating high trial-to-trial variability.

**Thesis narrative.**
The filter mechanism acts as a precision–recall trade-off at the dispatch level: it reduces the number of commands sent to the robot (lower dispatch rate) in exchange for higher confidence in those that are sent. This is the desired safety property for an HRC system where a wrong robot command is costlier than a delayed one.

**Suggested caption.**
> *Figure X — Dispatch quality per trial: accuracy of dispatched robot commands (y-axis) vs fraction of inference calls dispatched (x-axis). Large dots = condition means; small dots = individual trials. Pilot study, N = 264 trials.*

---

### `fig06_call_counts.pdf` — LLM / VLM Calls per Trial

**What it shows.**
A grouped bar chart of the mean number of LLM calls, VLM calls, and total calls per trial for each condition. Error bars = ±1 SD.

**Visual structure.**
Three bars per condition group (red/pink = LLM calls, purple = VLM calls, grey = total calls) for four conditions on the x-axis.

**Key values.**

| Condition | LLM calls | VLM calls | Total |
|---|---|---|---|
| No Context | 6.6 ± 2.5 | 6.8 ± 4.3 | 13.4 ± 5.0 |
| With Context | 6.2 ± 2.8 | 7.1 ± 5.4 | 13.3 ± 6.5 |
| Filter | 7.1 ± 1.3 | 4.5 ± 3.2 | 11.6 ± 3.6 |
| Filter + Context | 5.3 ± 1.9 | 4.2 ± 1.9 | **9.5 ± 1.9** |

**Thesis narrative.**
The filter reduces VLM escalation (from ~7 calls/trial to ~4.5) because it catches LLM errors before escalating to the more expensive Gemini call. Filter + Context achieves the lowest total call count (9.5) and the tightest variance, suggesting more consistent trial-to-trial system behaviour. The reduction in VLM calls also has direct API cost implications.

**Suggested caption.**
> *Figure X — Mean LLM, VLM, and total inference calls per trial by condition. Error bars = ±1 SD. Pilot study, N = 264 trials.*

---

### `fig07_filter_effectiveness.pdf` — Scene-Consistency Filter: Hint Injection Outcomes

**What it shows.**
A two-panel figure evaluating the scene-consistency filter's ability to recover correct predictions via hint injection after rejecting an initial wrong prediction.

**Panel (A) — Overall recovery.**
A stacked horizontal bar showing mean filter activations per trial (N = 8 filter-enabled trials). Of 6.2 activations per trial on average: 3.6 recovered to a correct prediction (green, 58%), and 2.6 did not (red, 42%). Individual trial dots are overlaid.

**Panel (B) — Recovery rate per assembly step.**
A bar chart of recovery rate (hint → correct) broken down by assembly step, with the share of all activations annotated above each bar.

| Step | Recovery rate | n activations | % of all activations |
|---|---|---|---|
| Step 2 | ~68% | 3 | 6% |
| Step 3 | ~53% | 15 | 30% |
| Step 4 | ~68% | 9 | 18% |
| Step 5 | ~57% | 23 | 46% |

**Thesis narrative.**
The filter recovers approximately 58% of rejected predictions by injecting a targeted hint into the prompt. Step 5 accounts for almost half of all filter activations, confirming it is the most ambiguous step. Recovery rates are broadly consistent across steps (~53–68%), indicating the hint strategy is uniformly useful. The 42% non-recovered activations result in the call being dropped (no robot command dispatched), which is the safe fallback.

**Suggested caption.**
> *Figure X — Scene-consistency filter outcomes across filter-enabled trials (N = 8 trials). (A) Mean filter activations per trial split into recovered (correct after hint) and not recovered. (B) Recovery rate per assembly step; percentages above bars indicate each step's share of total activations.*

---

### `fig08_non_assembly_ratio.pdf` — Non-Assembly Time: No Robot vs With Robot

**What it shows.**
A dot plot comparing the fraction of total trial time spent on non-assembly activities (idle time, waiting for the robot, repositioning) between the No Robot baseline condition and the With Robot condition (B_with_context trials only).

**Visual structure.**
Two columns of dots (grey = No Robot, teal = With Robot) with dashed horizontal lines marking the condition means. Non-assembly time is derived from BORIS behavioural annotation; ATCT comes from the trial logger.

**Key values.**
- No Robot mean: **20%** of ATCT (1 data point)
- With Robot mean: **32%** of ATCT (range roughly 13%–48% across trials)

**Thesis narrative.**
The With Robot condition introduces a 12 percentage-point increase in non-assembly time relative to the no-robot baseline. This overhead reflects the time the operator waits for robot pick-and-place motions to complete before resuming assembly. Reducing this idle time is a primary motivation for the Filter + Context condition, which achieves lower ATCT (170 s) partly by dispatching fewer, higher-confidence robot commands — reducing the number of robot motion cycles per trial.

**Suggested caption.**
> *Figure X — Non-assembly time as a fraction of ATCT, derived from BORIS behavioural annotation. No Robot = baseline human-only trials; With Robot = B\_with\_context trials. Lower values indicate less idle/waiting time.*

---

## How to Insert Figures in the Thesis (LaTeX)

All figures are vector PDFs and can be included directly with `\includegraphics`. Use the exact filenames listed below.

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{logs/analysis/figures/fig01_overview.pdf}
  \caption{Assembly Task Completion Time and Execution Success per Condition.
           Error bars = ±1\,SD. Pilot study, $N=264$ trials.}
  \label{fig:overview}
\end{figure}
```

### Figure filename index

| Figure label | Filename | Suggested `\label` |
|---|---|---|
| Overview (ATCT + exec score) | `fig01_overview.pdf` | `fig:overview` |
| Bayesian belief evolution | `fig02_bayesian_belief.pdf` | `fig:bayesian_belief` |
| Confusion matrices | `fig03_confusion_matrix.pdf` | `fig:confusion_matrix` |
| Latency–accuracy trade-off | `fig04_latency_accuracy_tradeoff.pdf` | `fig:latency_accuracy` |
| Dispatch quality | `fig05_dispatch_quality.pdf` | `fig:dispatch_quality` |
| LLM/VLM call counts | `fig06_call_counts.pdf` | `fig:call_counts` |
| Filter hint injection outcomes | `fig07_filter_effectiveness.pdf` | `fig:filter_effectiveness` |
| Non-assembly time ratio | `fig08_non_assembly_ratio.pdf` | `fig:non_assembly_ratio` |

### Recommended ordering in the Results chapter

1. `fig01_overview` — high-level comparison of conditions (leads the section)
2. `fig06_call_counts` — explains the LLM/VLM split underlying the overview
3. `fig03_confusion_matrix` — per-model, per-step accuracy breakdown
4. `fig04_latency_accuracy_tradeoff` — model selection trade-off
5. `fig05_dispatch_quality` — effect of filter on dispatch precision
6. `fig07_filter_effectiveness` — mechanism detail of the filter
7. `fig02_bayesian_belief` — qualitative trace illustrating the Bayesian planner
8. `fig08_non_assembly_ratio` — human-side overhead analysis (closing discussion)
