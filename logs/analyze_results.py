#!/usr/bin/env python3
"""
analyze_results.py  —  HRC Thesis / Conference Paper
=======================================================
Publication-quality figure generation for the robotic assembly
assistance system evaluation.

Usage
-----
    python analyze_results.py [--log-dir logs] [--output-dir output]

Outputs
-------
    output/figures/fig01_accuracy_context.pdf   ... fig10_fluency_breakdown.pdf
    output/metrics_summary.xlsx  (sheets: metrics_summary, step_accuracy)
    output/pilot_report.txt
"""

import argparse
import json
import math
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless backend — no display required
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

N_STEPS = 5

CONDITIONS = ["A_no_robot", "B_no_context", "B_with_context",
              "C_with_filter", "C_with_filter_and_context"]

CONDITION_LABELS = {
    "A_no_robot":                "No Robot",
    "B_no_context":              "No Context",
    "B_with_context":            "With Context",
    "C_with_filter":             "Filter",
    "C_with_filter_and_context": "Filter + Context",
}

# Okabe-Ito colorblind-safe palette for conditions.
# All five condition colors are perceptually distinct and safe for deuteranopia/protanopia.
PALETTE = {
    "A_no_robot":                "#999999",  # grey   — baseline / no robot
    "B_no_context":              "#E69F00",  # amber  — robot, no context
    "B_with_context":            "#009E73",  # teal   — robot, with context
    "C_with_filter":             "#0072B2",  # blue   — filter, no context
    "C_with_filter_and_context": "#CC79A7",  # pink   — filter + context
    "llm":                       "#D65F5F",
    "vlm":                       "#B47CC7",
    "hf_api":                    "#E8B86D",
    "local":                     "#D65F5F",
}

# Conditions that involve LLM/VLM inference (not A_no_robot)
INFERENCE_CONDITIONS = ["B_no_context", "B_with_context",
                        "C_with_filter", "C_with_filter_and_context"]

# Conditions that use the scene-consistency filter
FILTER_CONDITIONS = {"C_with_filter", "C_with_filter_and_context"}

STEP_COLORS = ["#82e0aa", "#7fb3d3", "#f0b27a", "#c39bd3", "#85c1e9"]

SINGLE_COL = 3.5    # inches — IEEE single column
DOUBLE_COL = 7.2    # inches — IEEE double column
DPI        = 300

# Stopwords for text->step resolution
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "on", "in", "at", "with",
    "for", "by", "onto", "into", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "from",
}


def _apply_rc():
    """Apply global matplotlib style for publication-quality output."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "figure.dpi":        DPI,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype":      42,    # embed fonts in PDF
    })


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Ingests all *_summary.json and *_predictions.jsonl files from log_dir.

    Produces two DataFrames:
      summaries_df  — one row per trial
      predictions_df — one row per LLM/VLM call
    """

    def __init__(self, log_dir: str, memory_path: str | None = None):
        self.log_dir = Path(log_dir)
        self.memory = self._load_memory(memory_path)
        self.summaries_df, self.predictions_df, self.filter_events_df = self._load_all()

    # ── memory ────────────────────────────────────────────────────────────────

    def _load_memory(self, explicit_path: str | None) -> list[dict]:
        candidates = []
        if explicit_path:
            candidates.append(Path(explicit_path))
        # learned_memory.json is the file used by llm.py and annotate_predictions.py
        # at runtime — it must take priority over the hand-edited memory.json.
        for name in ("learned_memory.json", "memory.json"):
            candidates.append(self.log_dir / name)
            candidates.append(self.log_dir.parent / name)
        for p in candidates:
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                print(f"  [DataLoader] Memory loaded from {p}")
                return data
        print("  [DataLoader] WARNING: no memory.json found — step resolution disabled")
        return []

    # ── main loader ───────────────────────────────────────────────────────────

    def _load_all(self):
        summary_files = sorted(self.log_dir.glob("*_summary.json"))
        if not summary_files:
            raise FileNotFoundError(
                f"No *_summary.json files found in {self.log_dir}"
            )

        sum_rows     = []
        pred_rows    = []
        filter_rows  = []

        for sf in summary_files:
            with open(sf, encoding="utf-8") as f:
                s = json.load(f)

            score = self._normalize_score(s.get("execution_success_score"), sf.name)
            participant_id = s.get("trial_id", "?")
            trial_number   = s.get("trial_number", 0)
            condition      = s.get("condition", "unknown")
            llm_backend    = s.get("llm_backend", "hf_api")

            if condition not in CONDITIONS:
                print(f"  [DataLoader] WARNING: unknown condition '{condition}' in {sf.name}")

            sum_rows.append({
                "participant_id":          participant_id,
                "trial_id":                trial_number,
                "trial_tag":               f"T{trial_number:02d}",
                "condition":               condition,
                "atct_seconds":            s.get("atct_seconds"),
                "vlm_escalation_rate":     s.get("vlm_escalation_rate"),
                "execution_success_score": score,
                "llm_call_count":          s.get("llm_call_count", 0),
                "vlm_call_count":          s.get("vlm_call_count", 0),
                "total_call_count":        s.get("total_call_count", 0),
                "llm_backend":             llm_backend,
                "llm_mean_latency_s":      s.get("llm_mean_latency_s"),
                "llm_std_latency_s":       s.get("llm_std_latency_s"),
                "vlm_mean_latency_s":      s.get("vlm_mean_latency_s"),
                "vlm_std_latency_s":       s.get("vlm_std_latency_s"),
                "_summary_path":           str(sf),
                # Scene-consistency filter counters (0 when filter was not enabled)
                "filter_enabled":              s.get("filter_enabled", False),
                "validation_failures_llm":     s.get("validation_failures_llm", 0),
                "validation_failures_vlm":     s.get("validation_failures_vlm", 0),
                "validation_failures_total":   s.get("validation_failures_total", 0),
                "retries_triggered_llm":       s.get("retries_triggered_llm", 0),
                "retries_triggered_vlm":       s.get("retries_triggered_vlm", 0),
                "hint_converged_valid_llm":    s.get("hint_converged_valid_llm", 0),
                "hint_converged_valid_vlm":    s.get("hint_converged_valid_vlm", 0),
                "hint_converged_none_llm":     s.get("hint_converged_none_llm", 0),
                "hint_converged_none_vlm":     s.get("hint_converged_none_vlm", 0),
                "drops_after_hint_llm":        s.get("drops_after_hint_llm", 0),
                "drops_after_hint_vlm":        s.get("drops_after_hint_vlm", 0),
            })

            jsonl_path = Path(str(sf).replace("_summary.json", "_predictions.jsonl"))
            if jsonl_path.exists():
                trial_preds, trial_filters = self._load_jsonl(
                    jsonl_path, participant_id, trial_number, condition, llm_backend
                )
                pred_rows.extend(trial_preds)
                filter_rows.extend(trial_filters)
            else:
                print(f"  [DataLoader] WARNING: no predictions JSONL for {sf.name}")

        summaries_df    = pd.DataFrame(sum_rows)
        predictions_df  = pd.DataFrame(pred_rows) if pred_rows else pd.DataFrame()
        filter_events_df = pd.DataFrame(filter_rows) if filter_rows else pd.DataFrame()
        return summaries_df, predictions_df, filter_events_df

    @staticmethod
    def _normalize_score(score, fname: str) -> float | None:
        """Return execution_success_score as a float in 0–1; no rescaling needed."""
        if score is None:
            return None
        return float(score)

    @staticmethod
    def _load_jsonl(path: Path, participant_id, trial_id, condition,
                    llm_backend) -> tuple[list[dict], list[dict]]:
        """Return (pred_rows, filter_event_rows). Dispatch events are used
        internally to build was_dispatched on pred_rows."""
        pred_rows     = []
        filter_events = []
        dispatch_events = []
        seq_idx = 0   # counts only prediction records (not events)

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = r.get("event_type")

                if event_type == "dispatch":
                    dispatch_events.append(r)
                    continue

                if event_type == "validation_failure":
                    filter_events.append({
                        **r,
                        "participant_id": participant_id,
                        "trial_id":       trial_id,
                        "condition":      condition,
                    })
                    continue

                if event_type is not None:
                    continue  # unknown future event type — skip

                # Normal prediction record
                gt    = r.get("ground_truth_step")
                pc    = r.get("prediction_correct")
                inp   = r.get("input", {})
                out   = r.get("output", {})
                probs = inp.get("step_probabilities") or {}

                if gt is None:
                    print(f"  [DataLoader] WARNING: unannotated call "
                          f"{r.get('call_id')} in {path.name}")

                pred_rows.append({
                    "participant_id":      participant_id,
                    "trial_id":            trial_id,
                    "trial_tag":           f"T{trial_id:02d}",
                    "condition":           condition,
                    "llm_backend":         llm_backend,
                    "call_id":             r.get("call_id"),
                    "call_sequence_index": seq_idx,
                    "type":                r.get("type"),
                    "timestamp":           r.get("timestamp"),
                    "inference_time_s":    r.get("inference_time_s"),
                    "semantic_action":     inp.get("semantic_action", ""),
                    "context_available":   inp.get("context_available", False),
                    "step_probabilities":  probs,
                    "stage_of_assembly":   out.get("stage_of_assembly", ""),
                    "next_operation":      out.get("next_operation", ""),
                    "ground_truth_step":   gt,
                    "prediction_correct":  pc,
                    "was_hint_injection":  r.get("was_hint_injection", False),
                })
                seq_idx += 1

        # Join dispatch events back to prediction rows
        dispatched_ids = {ev.get("dispatched_from_call_id") for ev in dispatch_events}
        for row in pred_rows:
            row["was_dispatched"] = row["call_id"] in dispatched_ids

        return pred_rows, filter_events


# ─────────────────────────────────────────────────────────────────────────────
# METRICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class MetricsEngine:
    """
    Derives all computed columns from the raw DataFrames produced by DataLoader.

    Adds to predictions_df:
        is_correct, max_prior_prob, prior_entropy, normalized_entropy,
        prior_correct_prob, predicted_step, normalized_call_pos

    Adds to summaries_df (joined from predictions_df):
        llm_accuracy, vlm_accuracy, overall_accuracy, unclear_rate

    Creates step_accuracy_df:
        step, type, accuracy, n_calls
    """

    def __init__(self, summaries_df: pd.DataFrame,
                 predictions_df: pd.DataFrame,
                 memory: list[dict],
                 filter_events_df: pd.DataFrame | None = None):
        self.memory = memory
        self.summaries_df  = summaries_df.copy()
        self.predictions_df = predictions_df.copy() if not predictions_df.empty \
                              else pd.DataFrame()
        self.filter_events_df = (filter_events_df.copy()
                                 if filter_events_df is not None and not filter_events_df.empty
                                 else pd.DataFrame())
        self.step_accuracy_df = pd.DataFrame()

        self._enrich()

    # ── enrichment ────────────────────────────────────────────────────────────

    def _enrich(self):
        if self.predictions_df.empty:
            return

        df = self.predictions_df

        # is_correct: True / False / NaN
        df["is_correct"] = df["prediction_correct"].apply(self._to_bool)

        # Bayesian prior metrics
        df["max_prior_prob"]    = df["step_probabilities"].apply(
            lambda d: max(d.values()) if d else float("nan")
        )
        df["prior_entropy"]     = df["step_probabilities"].apply(self._entropy)
        df["normalized_entropy"]= df["prior_entropy"].apply(
            lambda h: h / math.log2(N_STEPS) if (h == h) else float("nan")   # nan-safe
        )
        df["prior_correct_prob"]= df.apply(self._prior_correct_prob, axis=1)

        # Resolve predicted step from stage_of_assembly text
        df["predicted_step"] = df["stage_of_assembly"].apply(self._resolve_step)

        # Ground truth step as integer (None for "none" / unannotated)
        df["gt_step_int"] = df["ground_truth_step"].apply(self._to_step_int)

        # Normalized call position within trial (0 = first, 1 = last)
        for trial_id, grp in df.groupby("trial_id"):
            n = len(grp)
            pos = grp["call_sequence_index"].values
            if n > 1:
                norm_pos = (pos - pos.min()) / max(pos.max() - pos.min(), 1)
            else:
                norm_pos = np.array([0.5])
            df.loc[grp.index, "normalized_call_pos"] = norm_pos

        self.predictions_df = df

        # Per-trial accuracy -> join to summaries
        self._compute_trial_accuracy()

        # Per-step accuracy DataFrame
        self._compute_step_accuracy()

    # ── per-trial accuracy ────────────────────────────────────────────────────

    def _compute_trial_accuracy(self):
        acc_rows = []
        for _, trial in self.summaries_df.iterrows():
            tid = trial["trial_id"]
            tp  = self.predictions_df[self.predictions_df["trial_id"] == tid]

            def _acc(mask):
                sub = tp[mask & tp["is_correct"].notna()]
                return sub["is_correct"].mean() if len(sub) > 0 else float("nan")

            unclear_count = (tp["prediction_correct"] == "unclear").sum()
            acc_rows.append({
                "trial_id":        tid,
                "llm_accuracy":    _acc(tp["type"] == "llm"),
                "vlm_accuracy":    _acc(tp["type"] == "vlm"),
                "overall_accuracy":_acc(pd.Series([True] * len(tp), index=tp.index)),
                "unclear_rate":    unclear_count / len(tp) if len(tp) > 0 else float("nan"),
            })

        acc_df = pd.DataFrame(acc_rows)
        self.summaries_df = self.summaries_df.merge(acc_df, on="trial_id", how="left")

    # ── per-step accuracy ─────────────────────────────────────────────────────

    def _compute_step_accuracy(self):
        df    = self.predictions_df
        valid = df[
            df["gt_step_int"].notna() &
            df["is_correct"].notna()
        ].copy()

        rows = []
        for step in range(1, N_STEPS + 1):
            for call_type in ("llm", "vlm"):
                sub = valid[
                    (valid["gt_step_int"] == step) & (valid["type"] == call_type)
                ]
                rows.append({
                    "step":     step,
                    "type":     call_type,
                    "accuracy": sub["is_correct"].mean() if len(sub) > 0 else float("nan"),
                    "n_calls":  len(sub),
                })
        self.step_accuracy_df = pd.DataFrame(rows)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_bool(v) -> float:
        if v is True or v == "True":
            return True
        if v is False or v == "False":
            return False
        return float("nan")

    @staticmethod
    def _to_step_int(v):
        if v is None or v == "none":
            return None
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _entropy(prob_dict: dict) -> float:
        if not prob_dict:
            return float("nan")
        vals = [v for v in prob_dict.values() if v > 0]
        s = sum(vals)
        if s == 0:
            return float("nan")
        return -sum((v / s) * math.log2(v / s) for v in vals)

    def _prior_correct_prob(self, row) -> float:
        gt    = row["ground_truth_step"]
        probs = row["step_probabilities"]
        if not probs or gt is None or gt == "none":
            return float("nan")
        try:
            return probs.get(f"step_{int(float(gt))}", float("nan"))
        except (ValueError, TypeError):
            return float("nan")

    def _resolve_step(self, text: str):
        """Match stage_of_assembly text to a step number via word overlap."""
        if not text or not self.memory:
            return None
        words = {w for w in text.lower().split() if w not in _STOPWORDS}
        best_step, best_score = None, 0
        for step in self.memory:
            step_words = {w for w in step["step description"].lower().split()
                          if w not in _STOPWORDS}
            score = len(words & step_words)
            if score > best_score:
                best_score = score
                best_step  = step["step number"]
        return best_step if best_score >= 1 else None


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

class FigureFactory:
    """One public method per figure (fig01 … fig12)."""

    def __init__(self, metrics: MetricsEngine, output_dir: Path):
        self.m          = metrics
        self.s          = metrics.summaries_df
        self.p          = metrics.predictions_df
        self.step_acc   = metrics.step_accuracy_df
        self.fe         = metrics.filter_events_df   # filter event rows
        self.memory     = metrics.memory
        self.out        = output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        _apply_rc()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _save(self, fig, name: str):
        path = self.out / name
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"    -> {path.name}")

    def _pilot_note(self, ax, n: int | None = None, extra: str = ""):
        n_trials = n if n is not None else len(self.s)
        note = f"Pilot study (N={n_trials} trials, 1 participant). Values are descriptive."
        if extra:
            note += f"  {extra}"
        ax.annotate(note, xy=(0.0, -0.20), xycoords="axes fraction",
                    fontsize=5.5, color="#888888", ha="left")

    def _cond_color(self, cond: str) -> str:
        return PALETTE.get(cond, "#999999")

    @staticmethod
    def _model_label(backend: str) -> str:
        """Map llm_backend value to a human-readable model name."""
        return {"hf_api": "Llama 4 Scout", "local": "Qwen3 4B INT8"}.get(backend, backend)

    @staticmethod
    def _safe_strip_labels(ax, xs, ys, labels, fontsize=5.5, x_offset=0.10,
                           color="#555", max_labels=12):
        """
        Annotate strip-plot dots with trial tags.
        Shows tags only when N <= max_labels to avoid clutter in large datasets.
        When shown, stagger the y-offset for closely spaced points.
        """
        if len(labels) > max_labels:
            return
        # Sort by y to detect proximity
        order = sorted(range(len(ys)), key=lambda i: ys[i])
        prev_y, prev_text_y = None, None
        for i in order:
            text_y = ys[i]
            if prev_y is not None and abs(ys[i] - prev_y) < (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03:
                # Too close — push this label up slightly relative to previous
                text_y = (prev_text_y or prev_y) + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
            ax.text(xs[i] + x_offset, text_y, labels[i],
                    fontsize=fontsize, va="center", color=color)
            prev_y = ys[i]
            prev_text_y = text_y

    # ── Fig 01 — Accuracy: with vs without context ────────────────────────────

    def fig01_accuracy_context(self):
        inference_conds = ["B_no_context", "B_with_context",
                           "C_with_filter", "C_with_filter_and_context"]

        # Three fixed series — always shown; bars collapse to 0 when no data.
        # (type, backend_filter, palette_key, legend_label)
        series = [
            ("llm", "hf_api", "hf_api", f"LLM ({self._model_label('hf_api')})"),
            ("llm", "local",  "local",  f"LLM ({self._model_label('local')})"),
            ("vlm", None,     "vlm",    "VLM (Gemini)"),
        ]

        n_series = len(series)
        width    = 0.20
        x        = np.arange(len(inference_conds))
        rng      = np.random.default_rng(0)

        fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.75, 3.4))

        for i, (ct, backend, pal_key, label) in enumerate(series):
            offset = (i - (n_series - 1) / 2) * (width + 0.02)
            means, errs, scatter_vals = [], [], []

            for cond in inference_conds:
                # Compute per-trial accuracy from predictions_df so we can
                # split LLM calls by backend (hf_api vs local).
                if self.p.empty:
                    vals = np.array([])
                else:
                    if ct == "vlm":
                        mask = (self.p["type"] == "vlm") & (self.p["condition"] == cond)
                    else:
                        mask = (
                            (self.p["type"] == "llm") &
                            (self.p["llm_backend"] == backend) &
                            (self.p["condition"] == cond)
                        )
                    sub = self.p[mask & self.p["is_correct"].notna()]
                    # One accuracy value per trial
                    if sub.empty:
                        vals = np.array([])
                    else:
                        vals = (
                            sub.groupby("trial_id")["is_correct"]
                               .mean()
                               .values
                               .astype(float)
                        )

                means.append(np.mean(vals) if len(vals) else np.nan)
                errs.append(np.std(vals, ddof=0) if len(vals) > 1 else 0.0)
                scatter_vals.append(vals)

            bar_heights = [0.0 if np.isnan(m) else m for m in means]
            ax.bar(x + offset, bar_heights, width=width,
                   color=PALETTE[pal_key], alpha=0.75, label=label,
                   zorder=3)

            for xi, (m, e) in enumerate(zip(means, errs)):
                if not np.isnan(m) and e > 0:
                    ax.errorbar(x[xi] + offset, m, yerr=e, fmt="none",
                                color="#333", lw=0.8, capsize=3, zorder=5)
                if np.isnan(m):
                    ax.text(x[xi] + offset, 0.03, "n/a",
                            ha="center", va="bottom", fontsize=5,
                            color="#aaa", rotation=90)

            # Raw trial dots
            for xi, vals in enumerate(scatter_vals):
                if len(vals):
                    jitter = rng.uniform(-0.05, 0.05, size=len(vals))
                    ax.scatter(x[xi] + offset + jitter, vals,
                               color="black", s=16, zorder=6, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in inference_conds],
                           fontsize=7.5, rotation=20, ha="right")
        ax.set_ylabel("Prediction Accuracy")
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.axhline(1.0, color="#ccc", lw=0.6, ls="--", zorder=1)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=6.5)
        ax.set_title("Prediction Accuracy by Condition")
        self._pilot_note(ax)
        fig.tight_layout()
        self._save(fig, "fig01_accuracy_context.pdf")

    # ── Fig 01 (NEW) — ATCT + Success Score ──────────────────────────────────

    def fig01_overview(self):
        """Single-panel figure: ATCT bars (left axis) + success score line (right axis)."""
        all_conds    = ["B_no_context", "B_with_context",
                        "C_with_filter", "C_with_filter_and_context"]
        cond_labels  = [CONDITION_LABELS[c] for c in all_conds]
        bar_w        = 0.50

        fig, ax_atct = plt.subplots(figsize=(DOUBLE_COL * 0.75, 3.8))
        ax_succ      = ax_atct.twinx()

        atct_vals, succ_vals = [], []
        for xi, cond in enumerate(all_conds):
            color = self._cond_color(cond)
            sub   = self.s[self.s["condition"] == cond]
            atct  = sub["atct_seconds"].dropna().values
            succ  = sub["execution_success_score"].dropna().values

            m_atct = float(np.nanmean(atct)) if len(atct) else np.nan
            m_succ = float(np.nanmean(succ)) if len(succ) else np.nan
            atct_vals.append(m_atct)
            succ_vals.append(m_succ)

            if not np.isnan(m_atct):
                ax_atct.bar(xi, m_atct, width=bar_w, color=color,
                            alpha=0.60, zorder=2, label=cond_labels[xi])
                std_a = float(np.nanstd(atct)) if len(atct) > 1 else 0.0
                if std_a > 0:
                    ax_atct.errorbar(xi, m_atct, yerr=std_a, fmt="none",
                                     color="#333", lw=0.8, capsize=4, zorder=4)
                ax_atct.text(xi, m_atct * 0.5, f"{m_atct:.0f} s",
                             ha="center", va="center", fontsize=7,
                             color="white", fontweight="bold")
            else:
                ax_atct.text(xi, 30, "pending", ha="center", va="center",
                             fontsize=7, color="#bbb", style="italic")

        # Success score diamonds on right axis
        xs_valid = [i for i, v in enumerate(succ_vals) if not np.isnan(v)]
        ys_valid = [v for v in succ_vals if not np.isnan(v)]
        if xs_valid:
            ax_succ.plot(xs_valid, ys_valid,
                         color="#2c3e50", lw=1.8, ls="--",
                         marker="D", ms=7, zorder=5, label="Success score",
                         alpha=0.45)
            for xi, sv in zip(xs_valid, ys_valid):
                ax_succ.text(xi - 0.30, sv + 0.07, f"{sv:.0%}",
                             fontsize=7, ha="right", va="bottom",
                             color="#2c3e50", fontweight="bold", alpha=0.85)

        ax_atct.set_xticks(range(len(all_conds)))
        ax_atct.set_xticklabels(cond_labels, fontsize=7.5, rotation=20, ha="right")
        ax_atct.set_ylabel("Assembly Task Completion Time (s)", fontsize=8)
        ax_succ.set_ylabel("Execution Success Score", fontsize=8)
        ax_succ.set_ylim(0, 1.35)
        ax_succ.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        atct_max = max((v for v in atct_vals if not np.isnan(v)), default=300)
        ax_atct.set_ylim(0, max(atct_max * 1.40, 250))
        ax_atct.set_xlim(-0.55, len(all_conds) - 0.45)

        # Combined legend
        bar_handles = [
            mpatches.Patch(color=self._cond_color(c), alpha=0.60,
                           label=CONDITION_LABELS[c])
            for c in all_conds
        ]
        line_handle = Line2D([0], [0], color="#2c3e50", lw=1.8, ls="--",
                             marker="D", ms=6, label="Success score", alpha=0.45)
        ax_atct.legend(handles=bar_handles + [line_handle],
                       loc="upper center", fontsize=6.5, framealpha=0.9,
                       ncol=5, bbox_to_anchor=(0.5, -0.24))

        ax_atct.set_title("Task Completion Time and Execution Success per Condition",
                          fontsize=9)
        self._pilot_note(ax_atct)
        fig.tight_layout()
        self._save(fig, "fig01_overview.pdf")

    # ── Fig 02 — Accuracy by step and call type ───────────────────────────────

    def fig02_accuracy_by_step(self):
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.2))
        steps      = list(range(1, N_STEPS + 1))
        call_types = [("llm", "LLM"), ("vlm", "VLM")]
        width = 0.33
        x     = np.arange(len(steps))

        for i, (ct, ct_label) in enumerate(call_types):
            offset = (i - (len(call_types) - 1) / 2) * width
            accs, ns = [], []
            for step in steps:
                row = self.step_acc[
                    (self.step_acc["step"] == step) & (self.step_acc["type"] == ct)
                ]
                accs.append(row["accuracy"].values[0] if len(row) else np.nan)
                ns.append(int(row["n_calls"].values[0]) if len(row) else 0)

            for xi, (acc, n) in enumerate(zip(accs, ns)):
                alpha = 0.4 if n < 3 else 0.82
                if not np.isnan(acc):
                    ax.bar(x[xi] + offset, acc, width=width,
                           color=PALETTE[ct], alpha=alpha, zorder=3,
                           label=ct_label if xi == 0 else "")
                    ax.text(x[xi] + offset, acc + 0.02, f"n={n}",
                            ha="center", fontsize=5.5, color="#444")
                else:
                    ax.bar(x[xi] + offset, 0, width=width, color="#ddd",
                           alpha=0.4, zorder=3)
                    ax.text(x[xi] + offset, 0.02, "—",
                            ha="center", fontsize=6, color="#bbb")

        def _step_label(s):
            if not self.memory or s > len(self.memory):
                return f"Step {s}"
            desc = self.memory[s - 1]["step description"]
            short = (desc[:22] + "…") if len(desc) > 22 else desc
            return f"Step {s}\n{short}"

        ax.set_xticks(x)
        ax.set_xticklabels([_step_label(s) for s in steps], fontsize=7)
        ax.set_ylabel("Prediction Accuracy")
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.axhline(1.0, color="#ccc", lw=0.6, ls="--", zorder=1)

        handles = [
            mpatches.Patch(color=PALETTE["llm"], label="LLM"),
            mpatches.Patch(color=PALETTE["vlm"], label="VLM"),
            mpatches.Patch(color="#ddd", alpha=0.6, label="n < 3 (low reliability)"),
        ]
        ax.legend(handles=handles, loc="upper right", framealpha=0.9)
        ax.set_title("Prediction Accuracy per Assembly Step and Model Type")
        fig.tight_layout()
        self._save(fig, "fig02_accuracy_by_step.pdf")

    # ── Fig 03 — ATCT per condition ───────────────────────────────────────────

    def fig03_atct_condition(self):
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.5))
        markers = ["o", "s", "^", "D", "v"]
        rng     = np.random.default_rng(1)

        for xi, cond in enumerate(CONDITIONS):
            sub = self.s[self.s["condition"] == cond]
            color = self._cond_color(cond)

            if len(sub) == 0:
                ax.text(xi, 80, "data\npending", ha="center", va="center",
                        fontsize=7, color="#bbb", style="italic",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ddd"))
                continue

            vals = sub["atct_seconds"].dropna().values
            tags = sub["trial_tag"].values
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))

            for j, v in enumerate(vals):
                ax.scatter(xi + jitter[j], v, s=55, color=color, zorder=5,
                           marker=markers[j % len(markers)], edgecolors="white", lw=0.5)

            mean_v = float(np.nanmean(vals))
            ax.hlines(mean_v, xi - 0.22, xi + 0.22,
                      colors=color, linewidth=2.0, linestyles="--", zorder=4)
            ax.text(xi + 0.24, mean_v, f"{mean_v:.0f} s",
                    va="center", fontsize=6.5, color=color, fontweight="bold")

        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=7.5)
        ax.set_ylabel("Task Completion Time (s)")
        ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)
        ymax = self.s["atct_seconds"].max() * 1.25 if not self.s.empty else 300
        ax.set_ylim(0, max(ymax, 250))
        ax.set_title("Assembly Task Completion Time per Condition")
        self._pilot_note(ax, extra="Condition A data pending.")
        fig.tight_layout()
        self._save(fig, "fig03_atct_condition.pdf")

    # ── Fig 04 — VLM escalation rate ─────────────────────────────────────────

    def fig02_vlm_escalation(self):
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.5))
        inference_conds = ["B_no_context", "B_with_context"]
        x_pos = {c: i for i, c in enumerate(inference_conds)}
        cmap  = plt.cm.RdYlGn
        rng   = np.random.default_rng(2)

        for cond in inference_conds:
            xi    = x_pos[cond]
            sub   = self.s[self.s["condition"] == cond]
            vals  = sub["vlm_escalation_rate"].dropna().values
            tags  = sub["trial_tag"].values
            scores= sub["execution_success_score"].fillna(0.5).values

            mean_v = float(np.nanmean(vals)) if len(vals) else np.nan
            if not np.isnan(mean_v):
                ax.bar(xi, mean_v, width=0.5, color=self._cond_color(cond),
                       alpha=0.45, zorder=2)

            jitters = rng.uniform(-0.04, 0.04, size=len(vals))
            xs_dot  = [xi + jitters[j] for j in range(len(vals))]
            for j, (v, sc) in enumerate(zip(vals, scores)):
                dot_color = cmap(float(sc))
                ax.scatter(xs_dot[j], v, s=75, color=dot_color,
                           zorder=5, edgecolors="#444", lw=0.6)

        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in inference_conds])
        ax.set_ylabel("VLM Escalation Rate")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

        sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.04)
        cbar.set_label("Execution Score", fontsize=6.5)
        cbar.ax.tick_params(labelsize=5.5)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["0%", "50%", "100%"])

        ax.set_title("VLM Escalation Rate per Condition")
        self._pilot_note(ax)
        fig.tight_layout()
        self._save(fig, "fig02_vlm_escalation.pdf")

    # ── Fig 02 (NEW NUMBER) — Latency distribution (no scatter dots) ─────────

    def fig02_latency_distribution(self):
        """Latency distribution as box-plots without individual scatter dots."""
        if self.p.empty:
            print("    [skip] no prediction data"); return

        categories = [
            ("Qwen3 4B INT8\n(local LLM)",  "llm", "local",  PALETTE["local"]),
            ("Llama 4 Scout\n(API LLM)",    "llm", "hf_api", PALETTE["hf_api"]),
            ("Gemini\n(VLM)",               "vlm", None,      PALETTE["vlm"]),
        ]
        box_data, cat_colors, cat_labels = [], [], []

        for label, ct, backend, color in categories:
            if ct == "vlm":
                mask = self.p["type"] == "vlm"
            elif backend:
                mask = (self.p["type"] == ct) & (self.p["llm_backend"] == backend)
            else:
                mask = self.p["type"] == ct
            vals = self.p.loc[mask, "inference_time_s"].dropna().values
            box_data.append(vals)
            cat_colors.append(color)
            cat_labels.append(label)

        fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.72, 4.2))
        positions = list(range(len(categories)))

        bp = ax.boxplot(box_data, positions=positions, widths=0.42,
                        patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8))

        for patch, color in zip(bp["boxes"], cat_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for i, (vals, color) in enumerate(zip(box_data, cat_colors)):
            if len(vals) == 0:
                ax.text(i, 0.3, "no data\n(pending)",
                        ha="center", va="center", fontsize=6, color="#aaa",
                        style="italic")
                continue
            med = float(np.median(vals))
            ax.text(i + 0.26, med, f"med={med:.1f}s",
                    ha="left", va="center", fontsize=5.5, color=color)

        ax.set_xticks(positions)
        ax.set_xticklabels(cat_labels, fontsize=8.0)
        ax.set_ylabel("Inference Time (s)")

        all_vals = [v for sub in box_data for v in sub]
        if all_vals:
            lo = min((v for v in all_vals if v > 0), default=0.1)
            if max(all_vals) / lo > 15:
                ax.set_yscale("log")
                ax.set_ylabel("Inference Time (s, log scale)")

        ax.set_title("Inference Latency by Model")
        self._pilot_note(ax, extra="Qwen3 4B INT8 (local): pending data if no local trials yet.")
        fig.tight_layout()
        self._save(fig, "fig02_latency_distribution.pdf")

    # ── Fig 05 — Latency distribution (legacy, kept for reference) ───────────

    def fig03_latency_distribution(self):
        """
        Compares inference latency across the three model roles:
          • Qwen (local LLM)  — fast, runs on local GPU
          • Llama 4 (HF API)  — slower, cloud API
          • Gemini (VLM)      — used for scene understanding escalation
        Reference line at 2 s marks the approximate human action cycle.
        """
        if self.p.empty:
            print("    [skip] no prediction data"); return

        categories = [
            ("Qwen3 4B INT8\n(local LLM)",   "llm", "local",  PALETTE["local"]),
            ("Llama 4 Scout\n(API LLM)",     "llm", "hf_api", PALETTE["hf_api"]),
            ("Gemini\n(VLM)",                "vlm", None,      PALETTE["vlm"]),
        ]
        box_data, cat_colors, cat_labels = [], [], []
        rng = np.random.default_rng(3)

        for label, ct, backend, color in categories:
            if ct == "vlm":
                mask = self.p["type"] == "vlm"
            elif backend:
                mask = (self.p["type"] == ct) & (self.p["llm_backend"] == backend)
            else:
                mask = self.p["type"] == ct
            vals = self.p.loc[mask, "inference_time_s"].dropna().values
            box_data.append(vals)
            cat_colors.append(color)
            cat_labels.append(label)

        fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.72, 4.2))
        positions = list(range(len(categories)))

        bp = ax.boxplot(box_data, positions=positions, widths=0.42,
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="o", markersize=2.5, alpha=0.4),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8))

        for patch, color in zip(bp["boxes"], cat_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for i, (vals, color) in enumerate(zip(box_data, cat_colors)):
            if len(vals) == 0:
                ax.text(i, 0.3, "no data\n(pending)",
                        ha="center", va="center", fontsize=6, color="#aaa",
                        style="italic")
                continue
            jitter = rng.uniform(-0.13, 0.13, size=len(vals))
            ax.scatter(i + jitter, vals, color=color, alpha=0.45, s=14, zorder=4)
            med = float(np.median(vals))
            ax.text(i + 0.26, med, f"med={med:.1f}s",
                    ha="left", va="center", fontsize=5.5, color=color)


        ax.set_xticks(positions)
        ax.set_xticklabels(cat_labels, fontsize=8.0)
        ax.set_ylabel("Inference Time (s)")

        all_vals = [v for sub in box_data for v in sub]
        if all_vals:
            lo = min((v for v in all_vals if v > 0), default=0.1)
            if max(all_vals) / lo > 15:
                ax.set_yscale("log")
                ax.set_ylabel("Inference Time (s, log scale)")

        ax.set_title("Inference Latency by Model")
        self._pilot_note(ax, extra="Qwen3 4B INT8 (local): pending data if no local trials yet.")
        fig.tight_layout()
        self._save(fig, "fig03_latency_distribution.pdf")

    # ── Fig 03 (NEW) — Bayesian belief heatmap ────────────────────────────────

    def fig03_bayesian_heatmap(self):
        """Bayesian belief evolution for the best P05 B_with_context trial.

        Layout (2 rows):
          Top   — step probability lines (one per assembly step) with GT step
                  regions shaded in the background.
          Bottom — per-call correctness markers (correct / wrong).

        Trial selection: P05, B_with_context, ranked by (GT step coverage,
        tracking accuracy, fewest calls) so the plot is clean and readable.
        """
        if self.p.empty:
            print("    [skip] no prediction data"); return

        TARGET_PID   = "P08"
        TARGET_TRIAL = 2

        def _safe_gt_int(g):
            """Return int step (1..N_STEPS) or 0 for missing/invalid."""
            try:
                if g is None or g != g:
                    return 0
                v = int(float(g))
                return v if 1 <= v <= N_STEPS else 0
            except (ValueError, TypeError):
                return 0

        def _truncate_after_step5(tp):
            """Drop everything after the last call annotated as step 5 (assembly done)."""
            mask = tp["gt_step_int"].apply(
                lambda g: _safe_gt_int(g) == N_STEPS
            )
            if not mask.any():
                return tp
            last5 = mask[mask].index[-1]
            return tp.loc[:last5].copy().reset_index(drop=True)

        trial_row = self.s[
            (self.s["participant_id"].astype(str) == TARGET_PID) &
            (self.s["trial_id"] == TARGET_TRIAL)
        ]
        if trial_row.empty:
            print(f"    [skip] trial {TARGET_PID} T{TARGET_TRIAL:02d} not found"); return

        trial_tag = f"T{TARGET_TRIAL:02d}"
        condition_label = trial_row.iloc[0]["condition"]

        raw_tp = self.p[
            (self.p["participant_id"].astype(str) == TARGET_PID) &
            (self.p["trial_id"] == TARGET_TRIAL) &
            self.p["step_probabilities"].apply(bool)
        ]
        tp = _truncate_after_step5(raw_tp.copy().reset_index(drop=True))

        if tp.empty:
            print(f"    [skip] no calls with step_probabilities in {TARGET_PID} {trial_tag}"); return

        call_idx = np.arange(len(tp))
        n_calls  = len(tp)

        # ── 3-panel layout: lines | GT strip | correctness ───────────────────
        fig, (ax_top, ax_gt, ax_bot) = plt.subplots(
            3, 1, figsize=(DOUBLE_COL, 4.8),
            gridspec_kw={"height_ratios": [5, 0.55, 0.75], "hspace": 0.05}
        )

        # ── Panel 1: Step probability lines ──────────────────────────────────
        for s_idx in range(N_STEPS):
            key   = f"step_{s_idx + 1}"
            probs = tp["step_probabilities"].apply(
                lambda d: float(d.get(key, 0.0))
            ).values
            ax_top.plot(call_idx, probs,
                        color=STEP_COLORS[s_idx], lw=1.8,
                        marker="o", ms=3.5, zorder=3,
                        label=f"Step {s_idx + 1}")
            ax_top.fill_between(call_idx, probs,
                                alpha=0.06, color=STEP_COLORS[s_idx])

        ax_top.set_xlim(-0.5, n_calls - 0.5)
        ax_top.set_ylim(0, 1.08)
        ax_top.set_ylabel("P(current step = i)", fontsize=8)
        ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax_top.tick_params(bottom=False, labelbottom=False)
        ax_top.legend(loc="lower right", fontsize=6.5, ncol=N_STEPS,
                      framealpha=0.9)

        # ── Panel 2: GT step strip + vertical dividers in ax_top ─────────────
        gt_ints_seq = [_safe_gt_int(g) for g in tp["gt_step_int"].tolist()]
        prev_gt, prev_i = None, 0
        transition_xs = []   # x positions of step boundaries
        for ci, gt_int in enumerate(gt_ints_seq):
            if gt_int != prev_gt:
                if prev_gt is not None and prev_gt > 0:
                    bx = prev_i - 0.5 if prev_i == 0 else ci - 0.5
                    ax_gt.axvspan(prev_i - 0.5, ci - 0.5,
                                  color=STEP_COLORS[prev_gt - 1], alpha=0.55,
                                  zorder=0)
                    if ci - prev_i >= 1:
                        ax_gt.text((prev_i + ci - 1) / 2, 0.5,
                                   f"S{prev_gt}",
                                   ha="center", va="center", fontsize=7,
                                   fontweight="bold", color="#333")
                    if prev_i > 0:   # skip the left edge
                        transition_xs.append(prev_i - 0.5)
                prev_gt, prev_i = gt_int, ci
        if prev_gt is not None and prev_gt > 0:
            ax_gt.axvspan(prev_i - 0.5, n_calls - 0.5,
                          color=STEP_COLORS[prev_gt - 1], alpha=0.55, zorder=0)
            ax_gt.text((prev_i + n_calls - 1) / 2, 0.5,
                       f"S{prev_gt}",
                       ha="center", va="center", fontsize=7,
                       fontweight="bold", color="#333")
            if prev_i > 0:
                transition_xs.append(prev_i - 0.5)

        # Draw dashed vertical lines at step boundaries in the probability panel
        for tx in transition_xs:
            ax_top.axvline(x=tx, color="#666", lw=0.9, ls="--",
                           zorder=2, alpha=0.6)

        ax_gt.set_xlim(-0.5, n_calls - 0.5)
        ax_gt.set_ylim(0, 1)
        ax_gt.set_yticks([])
        ax_gt.set_ylabel("GT", fontsize=7, labelpad=2)
        ax_gt.tick_params(bottom=False, labelbottom=False)
        for sp in ax_gt.spines.values():
            sp.set_visible(False)

        # ── Panel 3: Correctness markers ─────────────────────────────────────
        for ci, (_, row) in enumerate(tp.iterrows()):
            ic = row["is_correct"]
            if ic is True or ic == 1.0:
                ax_bot.scatter(ci, 0.5, marker="^", color="#27ae60",
                               s=28, zorder=5, edgecolors="white", lw=0.4)
            elif ic is False or ic == 0.0:
                ax_bot.scatter(ci, 0.5, marker="v", color="#e74c3c",
                               s=28, zorder=5, edgecolors="white", lw=0.4)

        ax_bot.set_xlim(-0.5, n_calls - 0.5)
        ax_bot.set_ylim(0, 1)
        ax_bot.set_yticks([])
        ax_bot.set_xlabel("Inference Call Index", fontsize=8)
        ax_bot.set_ylabel("Correct?", fontsize=7, labelpad=2)
        for sp in ax_bot.spines.values():
            sp.set_visible(False)

        bot_handles = [
            Line2D([0], [0], marker="^", lw=0, markerfacecolor="#27ae60",
                   markersize=7, label="Correct"),
            Line2D([0], [0], marker="v", lw=0, markerfacecolor="#e74c3c",
                   markersize=7, label="Wrong"),
        ]
        ax_bot.legend(handles=bot_handles, loc="upper right",
                      fontsize=6.5, ncol=2, framealpha=0.85)

        fig.suptitle(
            f"Bayesian Belief Evolution — {TARGET_PID} Trial {trial_tag}  ({condition_label})",
            fontsize=9, y=1.01
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.98])
        self._save(fig, "fig02_bayesian_belief.pdf")

    # ── Fig 06 — Bayesian belief evolution (legacy line chart) ───────────────

    def fig04_bayesian_evolution(self):
        if self.p.empty:
            print("    [skip] no prediction data"); return

        # Choose best trial for visualization. Goal: showcase that
        # B_with_context actually tracks the ground truth through the full
        # assembly, i.e. the belief argmax follows GT transitions. Ranking:
        # 0) HARD FILTER — trial must cover all 5 GT steps (otherwise the
        #    plot cannot show the belief walking through the full procedure).
        # 1) tracking_acc — fraction of calls whose argmax(step_probabilities)
        #    equals the ground-truth step. Direct measure of "context works".
        # 2) transition_resp — mean probability mass placed on the new GT
        #    step within RESP_WINDOW calls after each GT change-point.
        #    Rewards responsiveness over dwelling on a single step.
        # 3) mean prior_correct_prob — tiebreaker for overall confidence.
        # 4) n_prob — denser plots preferred as a final tiebreaker.
        cand = self.s[
            (self.s["condition"] == "B_with_context") &
            (self.s["participant_id"].astype(str).str.contains("P02", case=False))
        ].copy()
        if cand.empty:
            print("    [skip] no B_with_context trials for P02"); return

        REQUIRED_STEPS = set(range(1, N_STEPS + 1))
        RESP_WINDOW    = 2  # calls after each GT change-point

        def _trial_scores(tid):
            tp = self.p[
                (self.p["trial_id"] == tid) &
                self.p["step_probabilities"].apply(bool)
            ].sort_values("call_sequence_index").reset_index(drop=True)
            if tp.empty:
                return pd.Series({"_covers_all": False, "_track_acc": 0.0,
                                  "_trans_resp": 0.0, "_mean_pcp": 0.0,
                                  "_n_prob": 0})

            gt_ints = tp["gt_step_int"].tolist()
            probs   = tp["step_probabilities"].tolist()

            def _gt(v):
                if v is None:
                    return None
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return None
            gt_ints = [_gt(g) for g in gt_ints]

            covers_all = REQUIRED_STEPS.issubset(
                {g for g in gt_ints if g is not None}
            )

            # Tracking accuracy: argmax(step_probabilities) vs GT
            matches = total = 0
            for g, d in zip(gt_ints, probs):
                if g is None or not d:
                    continue
                total += 1
                argmax_key = max(d, key=lambda k: d[k])
                try:
                    pred = int(argmax_key.split("_")[-1])
                except (ValueError, IndexError):
                    continue
                if pred == g:
                    matches += 1
            track_acc = matches / total if total else 0.0

            # Transition responsiveness: probability on new GT right after
            # each change-point, averaged across all transitions.
            resp_scores = []
            prev = None
            for i, g in enumerate(gt_ints):
                if g is None:
                    continue
                if prev is not None and g != prev:
                    window = probs[i : i + RESP_WINDOW]
                    vals = [float(d.get(f"step_{g}", 0.0))
                            for d in window if d]
                    if vals:
                        resp_scores.append(sum(vals) / len(vals))
                prev = g
            trans_resp = (sum(resp_scores) / len(resp_scores)
                          if resp_scores else 0.0)

            mean_pcp = float(tp["prior_correct_prob"].dropna().mean() or 0.0)

            return pd.Series({
                "_covers_all": covers_all,
                "_track_acc":  track_acc,
                "_trans_resp": trans_resp,
                "_mean_pcp":   mean_pcp,
                "_n_prob":     int(len(tp)),
            })

        stats = cand["trial_id"].apply(_trial_scores)
        for col in ["_covers_all", "_track_acc", "_trans_resp",
                    "_mean_pcp", "_n_prob"]:
            cand[col] = stats[col].values

        full = cand[cand["_covers_all"]]
        if full.empty:
            print("    [warn] no B_with_context trial covers all 5 GT steps; "
                  "falling back to partial-coverage pool")
            pool = cand
        else:
            pool = full

        cand_sorted = pool.sort_values(
            ["_track_acc", "_trans_resp", "_mean_pcp", "_n_prob"],
            ascending=[False, False, False, False],
            na_position="last",
        )
        best_tid = cand_sorted.iloc[0]["trial_id"]
        trial_tag = f"T{int(best_tid):02d}"

        tp = self.p[
            (self.p["trial_id"] == best_tid) &
            self.p["step_probabilities"].apply(bool)
        ].copy().reset_index(drop=True)

        if tp.empty:
            print(f"    [skip] no calls with step_probabilities in {trial_tag}"); return

        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.8))
        call_idx = tp["call_sequence_index"].values

        # --- Five probability lines ---
        for s_idx in range(N_STEPS):
            key   = f"step_{s_idx + 1}"
            probs = tp["step_probabilities"].apply(
                lambda d: float(d.get(key, 0.0))
            ).values
            ax.plot(call_idx, probs, color=STEP_COLORS[s_idx],
                    lw=1.6, label=f"Step {s_idx + 1}",
                    marker="o", ms=3.5, zorder=3)
            ax.fill_between(call_idx, probs, alpha=0.07,
                            color=STEP_COLORS[s_idx])

        # --- Ground-truth change markers ---
        prev_gt = None
        for _, row in tp.iterrows():
            gt = row["ground_truth_step"]
            if gt == "none" or gt is None:
                continue
            try:
                gt_int = int(float(gt))
            except (ValueError, TypeError):
                continue
            if gt_int != prev_gt:
                ax.axvline(x=row["call_sequence_index"],
                           color="#777", lw=0.9, ls="--", zorder=2, alpha=0.7)
                ax.text(row["call_sequence_index"] + 0.1, 0.96,
                        f"GT: S{gt_int}", fontsize=6, va="top",
                        color="#555", transform=ax.get_xaxis_transform())
                prev_gt = gt_int

        # --- Correct / wrong triangle markers below x-axis ---
        for _, row in tp.iterrows():
            ic = row["is_correct"]
            if ic is True or ic == 1.0:
                color = "#27ae60"
            elif ic is False or ic == 0.0:
                color = "#e74c3c"
            else:
                continue
            ax.scatter(row["call_sequence_index"], -0.06,
                       marker="^", color=color, s=22, zorder=5,
                       clip_on=False, edgecolors="white", lw=0.3)

        ax.set_xlabel("Inference Call Index")
        ax.set_ylabel("P(current step = i)")
        ax.set_ylim(-0.10, 1.08)
        ax.set_xlim(float(call_idx.min()) - 0.5, float(call_idx.max()) + 0.5)

        # Combined legend: step lines + correctness markers
        step_handles = [
            Line2D([0], [0], color=STEP_COLORS[i], lw=1.6, marker="o", ms=4,
                   label=f"Step {i + 1}")
            for i in range(N_STEPS)
        ]
        marker_handles = [
            Line2D([0], [0], marker="^", lw=0, markerfacecolor="#27ae60",
                   markersize=7, label="Correct"),
            Line2D([0], [0], marker="^", lw=0, markerfacecolor="#e74c3c",
                   markersize=7, label="Wrong"),
        ]
        ax.legend(handles=step_handles + marker_handles,
                  loc="upper right", ncol=2, fontsize=6.5, framealpha=0.80)

        ax.set_title(
            f"Bayesian Belief Evolution — P02 Trial {trial_tag}  (B_with_context)"
        )
        self._pilot_note(ax, n=1, extra=f"Showing P02 trial {trial_tag} (best accuracy + GT coverage).")
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        self._save(fig, "fig04_bayesian_evolution.pdf")

    # ── Fig 07 — Confusion matrix ────────────────────────────────────────────

    def fig05_confusion_matrix(self):
        if self.p.empty:
            print("    [skip] no prediction data"); return

        valid = self.p[
            self.p["gt_step_int"].notna() &
            self.p["predicted_step"].notna() &
            self.p["is_correct"].notna()
        ].copy()
        valid["gt_int"]   = valid["gt_step_int"].astype(int)
        valid["pred_int"] = valid["predicted_step"].astype(int)

        # Build panels: one per (type, backend) combination actually present
        panels = []
        for ct, cmap_name in [("llm", "Blues"), ("vlm", "Purples")]:
            sub_ct = valid[valid["type"] == ct]
            if sub_ct.empty:
                continue
            if ct == "llm":
                for backend in sorted(sub_ct["llm_backend"].dropna().unique()):
                    sub_b = sub_ct[sub_ct["llm_backend"] == backend]
                    if not sub_b.empty:
                        model_name = self._model_label(backend)
                        panels.append((sub_b, cmap_name, f"LLM — {model_name}"))
            else:
                panels.append((sub_ct, cmap_name, "VLM — Gemini"))

        if not panels:
            print("    [skip] no annotated predictions"); return

        # Each panel should be square: allocate ~3.8 in per panel
        panel_w = 3.8
        fig_w   = panel_w * len(panels) + 0.6 * len(panels)   # +colorbar space
        fig, axes = plt.subplots(1, len(panels),
                                 figsize=(fig_w, panel_w + 0.6),
                                 squeeze=False)
        axes = axes[0]

        for ax, (sub, cmap_name, title) in zip(axes, panels):
            mat = np.zeros((N_STEPS, N_STEPS), dtype=int)
            for _, row in sub.iterrows():
                gi, pi = row["gt_int"] - 1, row["pred_int"] - 1
                if 0 <= gi < N_STEPS and 0 <= pi < N_STEPS:
                    mat[gi, pi] += 1

            row_sums = mat.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_mat = np.where(row_sums > 0, mat / row_sums, 0.0)

            im = ax.imshow(norm_mat, cmap=cmap_name, vmin=0, vmax=1, aspect="equal")

            for i in range(N_STEPS):
                for j in range(N_STEPS):
                    cnt   = mat[i, j]
                    total = int(row_sums[i, 0])
                    pct   = f"\n({cnt/total:.0%})" if total > 0 and cnt > 0 else ""
                    text  = f"{cnt}{pct}"
                    col   = "white" if norm_mat[i, j] > 0.55 else "black"
                    ax.text(j, i, text, ha="center", va="center",
                            fontsize=6.5, color=col)

            step_lbls = [f"S{i+1}" for i in range(N_STEPS)]
            ax.set_xticks(range(N_STEPS)); ax.set_xticklabels(step_lbls)
            ax.set_yticks(range(N_STEPS)); ax.set_yticklabels(step_lbls)
            ax.set_xlabel("Predicted Step", fontsize=8)
            ax.set_ylabel("Ground Truth Step", fontsize=8)
            ax.set_title(f"{title}  (n={len(sub)})")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Row-normalised", fontsize=5.5)
            cb.ax.tick_params(labelsize=5.5)

        fig.suptitle("Confusion Matrix: Predicted vs Ground Truth Step", fontsize=10)
        self._pilot_note(axes[0])
        fig.tight_layout()
        self._save(fig, "fig05_confusion_matrix.pdf")

    # ── Fig 06 — Dispatch accuracy vs dispatch rate ───────────────────────────

    def fig06_dispatch_quality(self):
        """Scatter: x = dispatch rate, y = accuracy of dispatched calls.
        Bottom panel: mean ± SD of filter activations per trial (C_with_filter only).
        """
        if self.p.empty:
            fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, 3.5))
            ax.text(0.5, 0.5, "No prediction data available.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#aaa", style="italic")
            ax.set_title("Dispatch Quality: Accuracy vs Dispatch Rate")
            self._save(fig, "fig05_dispatch_quality.pdf")
            return

        points = []
        for cond in INFERENCE_CONDITIONS:
            sub = self.p[self.p["condition"] == cond]
            if sub.empty:
                continue

            has_dispatch = "was_dispatched" in sub.columns

            if cond in FILTER_CONDITIONS and has_dispatch:
                dispatched = sub[sub["was_dispatched"] == True]
                n_total    = len(sub)
                n_disp     = len(dispatched)
                rate       = n_disp / n_total if n_total > 0 else float("nan")
                valid      = dispatched[dispatched["is_correct"].notna()]
                acc        = float(valid["is_correct"].mean()) if not valid.empty else float("nan")
            else:
                n_total = len(sub)
                rate    = 1.0 if n_total > 0 else float("nan")
                valid   = sub[sub["is_correct"].notna()]
                acc     = float(valid["is_correct"].mean()) if not valid.empty else float("nan")

            points.append({
                "cond":  cond,
                "label": CONDITION_LABELS[cond],
                "color": self._cond_color(cond),
                "rate":  rate,
                "acc":   acc,
                "n":     n_total,
            })

        # ── Per-trial individual points (computed with same logic as condition means) ──
        trial_pts = []
        has_dispatch_col = "was_dispatched" in self.p.columns
        for cond in INFERENCE_CONDITIONS:
            sub_cond = self.p[self.p["condition"] == cond]
            if sub_cond.empty:
                continue
            color = self._cond_color(cond)
            for tid in sub_cond["trial_id"].unique():
                sub_t = sub_cond[sub_cond["trial_id"] == tid]
                if cond in FILTER_CONDITIONS and has_dispatch_col:
                    disp_t  = sub_t[sub_t["was_dispatched"] == True]
                    rate_t  = len(disp_t) / len(sub_t) if len(sub_t) > 0 else float("nan")
                    valid_t = disp_t[disp_t["is_correct"].notna()]
                else:
                    rate_t  = 1.0 if len(sub_t) > 0 else float("nan")
                    valid_t = sub_t[sub_t["is_correct"].notna()]
                acc_t = float(valid_t["is_correct"].mean()) if not valid_t.empty else float("nan")
                if not (np.isnan(rate_t) or np.isnan(acc_t)):
                    trial_pts.append({"rate": rate_t, "acc": acc_t, "color": color})

        # ── Single-panel scatter ───────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, 3.8))

        # Trial-level dots: same color as condition, very transparent, no edge ring
        for tp in trial_pts:
            ax.scatter(tp["rate"], tp["acc"],
                       s=55, color=tp["color"], zorder=3,
                       alpha=0.18, edgecolors="none")

        # Condition means: opaque, with edge — the legend refers to these
        for pt in points:
            if np.isnan(pt["rate"]) or np.isnan(pt["acc"]):
                continue
            ax.scatter(pt["rate"], pt["acc"],
                       s=90, color=pt["color"], zorder=5,
                       edgecolors="#333", lw=0.8, label=pt["label"])

        missing = [pt for pt in points if np.isnan(pt["rate"]) or np.isnan(pt["acc"])]
        if missing:
            ax.text(0.98, 0.02,
                    "Pending: " + ", ".join(m["label"] for m in missing),
                    transform=ax.transAxes, fontsize=5.5, color="#bbb",
                    ha="right", va="bottom", style="italic")

        ax.set_xlabel("Dispatch Rate  (calls sent to robot / total)", fontsize=8)
        ax.set_ylabel("Dispatch Accuracy", fontsize=8)
        ax.set_xlim(-0.05, 1.15)
        ax.set_ylim(0, 1.15)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.axhline(1.0, color="#ddd", lw=0.6, ls="--", zorder=1)
        ax.axvline(1.0, color="#ddd", lw=0.6, ls="--", zorder=1)
        ax.legend(fontsize=7, framealpha=0.9, loc="lower right",
                  title="Condition", title_fontsize=7)
        ax.set_title("Dispatch Quality: Accuracy vs Dispatch Rate", fontsize=9)
        self._pilot_note(ax)
        fig.tight_layout()
        self._save(fig, "fig05_dispatch_quality.pdf")

    # ── Fig 08 — Execution success score ─────────────────────────────────────

    def fig06_success_score(self):
        fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.7, 4.2))
        rng = np.random.default_rng(4)

        ax.axhline(1.0, color="#bbb", lw=0.8, ls="--", zorder=1)
        ax.text(len(CONDITIONS) - 0.5, 1.06, "Perfect",
                fontsize=6, color="#999", ha="right")

        for xi, cond in enumerate(CONDITIONS):
            sub   = self.s[self.s["condition"] == cond]
            color = self._cond_color(cond)
            vals  = sub["execution_success_score"].dropna().values
            tags  = sub["trial_tag"].values

            mean_v = float(np.nanmean(vals)) if len(vals) else np.nan

            if not np.isnan(mean_v):
                ax.vlines(xi, 0, mean_v, colors=color, lw=2.5, zorder=2)
                ax.scatter(xi, mean_v, s=70, color=color, zorder=4,
                           edgecolors="white", lw=0.8)
                ax.text(xi - 0.12, mean_v, f"{mean_v:.0%}",
                        fontsize=6.5, va="center", ha="right",
                        color=color, fontweight="bold")

            jitters = rng.uniform(-0.06, 0.06, size=len(vals))
            xs_d = [xi + jitters[j] for j in range(len(vals))]
            for j, v in enumerate(vals):
                dot_color = "#e74c3c" if float(v) < 0.5 else color
                ax.scatter(xs_d[j], float(v), s=28, color=dot_color,
                           zorder=5, edgecolors="white", lw=0.5)

            if len(vals) == 0:
                ax.text(xi, 0.5, "pending", ha="center", va="center",
                        fontsize=7, color="#bbb", style="italic")

        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS],
                           fontsize=8, rotation=15, ha="right")
        ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
        ax.set_ylabel("Execution Score (0–1)")
        ax.set_ylim(0, 1.18)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title("Task Execution Success Score per Condition")
        self._pilot_note(ax, extra="Red dots: failed trials (< 50%).")
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        self._save(fig, "fig06_success_score.pdf")

    # ── Fig 09 — Prior entropy vs accuracy ───────────────────────────────────

    def fig09_entropy_accuracy(self):
        if self.p.empty:
            print("    [skip] no prediction data"); return
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.3))

        valid = self.p[
            self.p["normalized_entropy"].notna() &
            self.p["is_correct"].notna()
        ].copy()
        valid["is_correct_01"] = valid["is_correct"].astype(float)

        if valid.empty:
            ax.text(0.5, 0.5, "Insufficient data\n(no context calls annotated)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#aaa")
            self._save(fig, "fig09_entropy_accuracy.pdf"); return

        for ct, color in [("llm", PALETTE["llm"]), ("vlm", PALETTE["vlm"])]:
            sub = valid[valid["type"] == ct]
            if not sub.empty:
                ax.scatter(sub["normalized_entropy"], sub["is_correct_01"],
                           color=color, alpha=0.55, s=28, label=ct.upper(),
                           zorder=4, edgecolors="white", lw=0.3)

        # Logistic regression if enough data
        if len(valid) >= 10:
            try:
                from sklearn.linear_model import LogisticRegression
                X = valid["normalized_entropy"].values.reshape(-1, 1)
                y = valid["is_correct_01"].values
                lr = LogisticRegression(random_state=0).fit(X, y)
                xr = np.linspace(0, 1, 120).reshape(-1, 1)
                ax.plot(xr, lr.predict_proba(xr)[:, 1],
                        color="#2c3e50", lw=1.2, ls="--",
                        label="logistic fit", zorder=5)
            except ImportError:
                pass
        else:
            ax.text(0.5, 0.10,
                    f"logistic fit needs N≥10 (N={len(valid)})",
                    transform=ax.transAxes, fontsize=6,
                    ha="center", color="#888")

        ax.set_xlabel("Bayesian Prior Entropy (normalized, 0=certain, 1=max uncertain)")
        ax.set_ylabel("Prediction Correct")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.18, 1.18)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Wrong (0)", "Correct (1)"])
        ax.legend(loc="lower left", framealpha=0.9)
        ax.set_title("Prior Uncertainty vs Prediction Correctness")
        self._pilot_note(ax, extra="Validates tracker calibration.")
        fig.tight_layout()
        self._save(fig, "fig09_entropy_accuracy.pdf")

    # ── Fig 07 — Latency–accuracy trade-off ──────────────────────────────────

    def fig07_latency_accuracy_tradeoff(self):
        if self.p.empty:
            print("    [skip] no prediction data"); return
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.3))

        groups = []
        for ct in ("llm", "vlm"):
            for backend in ("local", "hf_api"):
                for cond in ("B_no_context", "B_with_context"):
                    if ct == "vlm" and backend == "local":
                        continue
                    if ct == "vlm":
                        mask = (self.p["type"] == "vlm") & (self.p["condition"] == cond)
                    else:
                        mask = ((self.p["type"] == "llm") &
                                (self.p["llm_backend"] == backend) &
                                (self.p["condition"] == cond))
                    sub = self.p[mask & self.p["is_correct"].notna()]
                    if sub.empty:
                        continue
                    model_name = ("Gemini" if ct == "vlm"
                                  else self._model_label(backend))
                    cond_short = CONDITION_LABELS.get(cond, cond)
                    groups.append({
                        "mean_lat": sub["inference_time_s"].mean(),
                        "accuracy": float(sub["is_correct"].mean()),
                        "n":        len(sub),
                        "label":    f"{model_name}\n({cond_short})",
                        "color":    PALETTE[ct],
                    })

        if not groups:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#aaa")
            self._save(fig, "fig07_latency_accuracy_tradeoff.pdf"); return

        max_lat = max(g["mean_lat"] for g in groups)

        # Plot scatter points
        for g in groups:
            size = max(30, min(200, g["n"] * 18))
            ax.scatter(g["mean_lat"], g["accuracy"], s=size,
                       color=g["color"], alpha=0.72, zorder=4,
                       edgecolors="#333", lw=0.5)

        # Spread labels vertically on the right side to avoid overlap.
        # Sort by accuracy, then push each label up by MIN_SEP if it would
        # collide with the previous one, then clamp the whole column down if
        # it overflows the top of the axes.
        MIN_SEP = 0.13
        sorted_groups = sorted(groups, key=lambda g: g["accuracy"])
        label_ys = [g["accuracy"] for g in sorted_groups]
        for i in range(1, len(label_ys)):
            if label_ys[i] < label_ys[i - 1] + MIN_SEP:
                label_ys[i] = label_ys[i - 1] + MIN_SEP
        y_top = 1.05
        if label_ys and label_ys[-1] > y_top:
            shift = label_ys[-1] - y_top
            label_ys = [y - shift for y in label_ys]
        label_x = max_lat * 1.15
        for g, ly in zip(sorted_groups, label_ys):
            ax.annotate(g["label"],
                        xy=(g["mean_lat"], g["accuracy"]),
                        xytext=(label_x, ly),
                        fontsize=5.5, color="#333", va="center",
                        arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.4))

        ax.set_xlabel("Mean Inference Latency (s)")
        ax.set_ylabel("Prediction Accuracy")
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlim(0, max_lat * 1.9)

        # Size legend
        for n_ref, label in [(5, "n=5"), (10, "n=10"), (20, "n=20")]:
            ax.scatter([], [], s=max(30, min(200, n_ref * 18)),
                       color="#bbb", label=label, alpha=0.6,
                       edgecolors="#888", lw=0.5)
        ax.legend(fontsize=6, framealpha=0.9, loc="lower right")
        ax.set_title("Latency–Accuracy Trade-off")
        self._pilot_note(ax)
        fig.tight_layout()
        self._save(fig, "fig07_latency_accuracy_tradeoff.pdf")

    # ── Fig 12 — Gantt-style decision timeline ────────────────────────────────

    def fig12_decision_timeline(self):
        if self.p.empty:
            print("    [skip] no prediction data"); return

        # Choose best B_with_context trial (highest success score)
        cand = self.s[self.s["condition"] == "B_with_context"]
        if cand.empty:
            cand = self.s[self.s["condition"] != "A_no_robot"]
        if cand.empty:
            print("    [skip] no suitable trial"); return

        best_tid  = cand.sort_values("execution_success_score",
                                     ascending=False).iloc[0]["trial_id"]
        trial_row = self.s[self.s["trial_id"] == best_tid].iloc[0]
        trial_tag = f"T{int(best_tid):02d}"

        tp = self.p[
            self.p["trial_id"] == best_tid
        ].copy().reset_index(drop=True)

        # Compute elapsed time from timestamps
        try:
            tp["ts"]        = pd.to_datetime(tp["timestamp"])
            t0              = tp["ts"].min()
            tp["elapsed_s"] = (tp["ts"] - t0).dt.total_seconds()
        except Exception:
            print(f"    [skip] timestamp parse error for {trial_tag}"); return

        total_time = float(tp["elapsed_s"].max()) + 8.0

        fig, axes = plt.subplots(
            2, 1, figsize=(DOUBLE_COL, 3.8),
            gridspec_kw={"height_ratios": [1, 2], "hspace": 0.05}
        )
        ax_gt, ax_calls = axes

        # ── Row 1: Ground-truth step bands ──────────────────────────────────
        segments, prev_step, prev_t = [], None, 0.0
        for _, row in tp.iterrows():
            gt = row["ground_truth_step"]
            t  = float(row["elapsed_s"])
            try:
                gt_int = int(float(gt)) if gt not in (None, "none") else 0
            except (ValueError, TypeError):
                gt_int = 0
            if gt_int != prev_step:
                if prev_step is not None:
                    segments.append((prev_step, prev_t, t))
                prev_step, prev_t = gt_int, t
        if prev_step is not None:
            segments.append((prev_step, prev_t, total_time))

        for step, t_s, t_e in segments:
            color = STEP_COLORS[step - 1] if 1 <= step <= N_STEPS else "#eeeeee"
            label = f"Step {step}" if step > 0 else "—"
            ax_gt.axvspan(t_s, t_e, color=color, alpha=0.5)
            ax_gt.text((t_s + t_e) / 2, 0.5, label,
                       ha="center", va="center", fontsize=7,
                       fontweight="bold", color="#333")

        ax_gt.set_xlim(0, total_time)
        ax_gt.set_ylim(0, 1)
        ax_gt.set_yticks([])
        ax_gt.set_ylabel("GT Step", fontsize=7, labelpad=2)
        atct = trial_row.get("atct_seconds")
        atct_str = f"{float(atct):.1f} s" if atct else "?"
        ax_gt.set_title(
            f"Decision Timeline — Trial {trial_tag}  "
            f"(condition: {trial_row['condition']},  ATCT = {atct_str})"
        )
        ax_gt.tick_params(bottom=False, labelbottom=False)
        for spine in ax_gt.spines.values():
            spine.set_visible(False)

        # ── Row 2: Inference calls ───────────────────────────────────────────
        for _, row in tp.iterrows():
            t    = float(row["elapsed_s"])
            ct   = row.get("type", "llm")
            color = PALETTE.get(ct, "#999")
            ic   = row["is_correct"]
            edge  = "#27ae60" if (ic is True or ic == 1.0) \
                    else "#e74c3c" if (ic is False or ic == 0.0) \
                    else "#aaa"

            # Vertical tick line
            ax_calls.vlines(t, 0.25, 0.75, colors=color, lw=1.6, zorder=3)

            # Call-type dot (LLM above centre, VLM below)
            y = 0.72 if ct == "vlm" else 0.28
            marker = "v" if ct == "vlm" else "^"
            ax_calls.scatter(t, y, s=32, color=color, marker=marker,
                             zorder=5, edgecolors=edge, lw=1.2)

            # Latency bar
            lat = float(row.get("inference_time_s") or 0)
            ax_calls.barh(0.5, lat, left=t, height=0.07,
                          color=color, alpha=0.28, zorder=2)

        ax_calls.set_xlim(0, total_time)
        ax_calls.set_ylim(0, 1)
        ax_calls.set_yticks([0.28, 0.72])
        ax_calls.set_yticklabels(["LLM", "VLM"], fontsize=7)
        ax_calls.set_xlabel("Elapsed Time (s)")
        ax_calls.set_ylabel("Call\nType", fontsize=7, labelpad=2)
        ax_calls.tick_params(left=False)

        # Robot activity footnote
        ax_calls.text(0.01, 0.04,
                      "Robot fetch activity not yet logged (fetch_time_s pending).",
                      transform=ax_calls.transAxes, fontsize=5.5,
                      color="#bbb", style="italic")

        # Legend
        handles = [
            mpatches.Patch(color=PALETTE["llm"], label="LLM call"),
            mpatches.Patch(color=PALETTE["vlm"], label="VLM call"),
            Line2D([0], [0], marker="^", lw=0, markerfacecolor="w",
                   markeredgecolor="#27ae60", ms=7, label="Correct"),
            Line2D([0], [0], marker="^", lw=0, markerfacecolor="w",
                   markeredgecolor="#e74c3c", ms=7, label="Wrong"),
        ]
        ax_calls.legend(handles=handles, loc="upper right",
                        fontsize=6.5, framealpha=0.9)

        fig.tight_layout()
        self._save(fig, "fig12_decision_timeline.pdf")

    # ── Fig 10 — LLM / VLM call counts per condition ─────────────────────────

    def fig10_call_counts(self):
        """
        Per-condition strip+bar chart of LLM calls, VLM calls, and total calls
        per trial.  Only inference conditions (B_*) are shown since A_no_robot
        has no robot calls.
        """
        inference_conds = ["B_no_context", "B_with_context"]
        series = [
            ("llm_call_count",   PALETTE["llm"],  "LLM calls"),
            ("vlm_call_count",   PALETTE["vlm"],  "VLM calls"),
            ("total_call_count", "#555555",        "Total calls"),
        ]
        n_series = len(series)
        width    = 0.22
        x        = np.arange(len(inference_conds))
        rng      = np.random.default_rng(7)

        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, 3.6))

        for i, (col, color, label) in enumerate(series):
            offset = (i - (n_series - 1) / 2) * (width + 0.02)
            for xi, cond in enumerate(inference_conds):
                sub  = self.s[self.s["condition"] == cond]
                vals = sub[col].dropna().values.astype(float)
                tags = sub["trial_tag"].values

                mean_v = float(np.nanmean(vals)) if len(vals) else np.nan
                std_v  = float(np.nanstd(vals))  if len(vals) > 1 else 0.0

                ax.bar(x[xi] + offset, mean_v if not np.isnan(mean_v) else 0,
                       width=width, color=color, alpha=0.55,
                       label=label if xi == 0 else "_nolegend_", zorder=3)

                if not np.isnan(mean_v) and std_v > 0:
                    ax.errorbar(x[xi] + offset, mean_v, yerr=std_v,
                                fmt="none", color="#333", lw=0.8,
                                capsize=3, zorder=5)

                if len(vals):
                    jitter = rng.uniform(-0.04, 0.04, size=len(vals))
                    xs_d   = x[xi] + offset + jitter
                    ax.scatter(xs_d, vals, s=28, color=color, zorder=6,
                               edgecolors="#333", lw=0.5, alpha=0.85)
                    self._safe_strip_labels(ax, xs_d.tolist(), vals.tolist(),
                                           tags.tolist(), fontsize=4.5,
                                           x_offset=0.03)

        ax.set_xticks(list(x))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in inference_conds])
        ax.set_ylabel("Calls per trial")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title("LLM / VLM Calls per Trial")
        self._pilot_note(ax)
        fig.tight_layout()
        self._save(fig, "fig10_call_counts.pdf")

    # ── Fig 04 (NEW NUMBER) — Confusion matrix (same content, new file name) ──

    def fig04_confusion_matrix(self):
        """Confusion matrix — same logic as fig05_confusion_matrix, saves as fig04."""
        if self.p.empty:
            print("    [skip] no prediction data"); return

        valid = self.p[
            self.p["gt_step_int"].notna() &
            self.p["predicted_step"].notna() &
            self.p["is_correct"].notna()
        ].copy()
        valid["gt_int"]   = valid["gt_step_int"].astype(int)
        valid["pred_int"] = valid["predicted_step"].astype(int)

        panels = []
        for ct, cmap_name in [("llm", "Blues"), ("vlm", "Purples")]:
            sub_ct = valid[valid["type"] == ct]
            if sub_ct.empty:
                continue
            if ct == "llm":
                for backend in sorted(sub_ct["llm_backend"].dropna().unique()):
                    sub_b = sub_ct[sub_ct["llm_backend"] == backend]
                    if not sub_b.empty:
                        panels.append((sub_b, cmap_name,
                                       f"LLM — {self._model_label(backend)}"))
            else:
                panels.append((sub_ct, cmap_name, "VLM — Gemini"))

        if not panels:
            print("    [skip] no annotated predictions"); return

        panel_w = 3.8
        fig_w   = panel_w * len(panels) + 0.6 * len(panels)
        fig, axes = plt.subplots(1, len(panels),
                                 figsize=(fig_w, panel_w + 0.6),
                                 squeeze=False)
        axes = axes[0]

        for ax, (sub, cmap_name, title) in zip(axes, panels):
            mat = np.zeros((N_STEPS, N_STEPS), dtype=int)
            for _, row in sub.iterrows():
                gi, pi = row["gt_int"] - 1, row["pred_int"] - 1
                if 0 <= gi < N_STEPS and 0 <= pi < N_STEPS:
                    mat[gi, pi] += 1

            row_sums = mat.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_mat = np.where(row_sums > 0, mat / row_sums, 0.0)

            im = ax.imshow(norm_mat, cmap=cmap_name, vmin=0, vmax=1, aspect="equal")

            for i in range(N_STEPS):
                for j in range(N_STEPS):
                    cnt   = mat[i, j]
                    total = int(row_sums[i, 0])
                    pct   = f"\n({cnt/total:.0%})" if total > 0 and cnt > 0 else ""
                    text  = f"{cnt}{pct}"
                    col   = "white" if norm_mat[i, j] > 0.55 else "black"
                    ax.text(j, i, text, ha="center", va="center",
                            fontsize=6.5, color=col)

            step_lbls = [f"S{i+1}" for i in range(N_STEPS)]
            ax.set_xticks(range(N_STEPS)); ax.set_xticklabels(step_lbls)
            ax.set_yticks(range(N_STEPS)); ax.set_yticklabels(step_lbls)
            ax.set_xlabel("Predicted Step", fontsize=8)
            ax.set_ylabel("Ground Truth Step", fontsize=8)
            ax.set_title(f"{title}  (n={len(sub)})")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Row-normalised", fontsize=5.5)
            cb.ax.tick_params(labelsize=5.5)

        fig.suptitle("Confusion Matrix: Predicted vs Ground Truth Step", fontsize=10)
        self._pilot_note(axes[0])
        fig.tight_layout()
        self._save(fig, "fig03_confusion_matrix.pdf")

    # ── Fig 05 (NEW NUMBER) — Latency–accuracy trade-off ────────────────────

    def fig05_latency_accuracy_tradeoff(self):
        """Latency–accuracy trade-off.

        Each (model, condition) combination is a dot. Color = model,
        marker shape = condition. Legend replaces all in-plot annotations.
        """
        if self.p.empty:
            print("    [skip] no prediction data"); return
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, 3.6))

        # Color per model, marker per condition
        MODEL_COLOR = {
            ("llm", "local"):  PALETTE["local"],
            ("llm", "hf_api"): PALETTE["hf_api"],
            ("vlm", None):     PALETTE["vlm"],
        }
        COND_MARKER = {
            "B_no_context":              "o",
            "B_with_context":            "s",
            "C_with_filter":             "^",
            "C_with_filter_and_context": "D",
        }

        groups = []
        for (ct, backend), color in MODEL_COLOR.items():
            for cond, marker in COND_MARKER.items():
                if ct == "vlm":
                    mask = (self.p["type"] == "vlm") & (self.p["condition"] == cond)
                else:
                    mask = ((self.p["type"] == "llm") &
                            (self.p["llm_backend"] == backend) &
                            (self.p["condition"] == cond))
                sub = self.p[mask & self.p["is_correct"].notna()]
                if sub.empty:
                    continue
                model_name = "Gemini" if ct == "vlm" else self._model_label(backend)

                lat_vals  = sub["inference_time_s"].dropna()
                mean_lat  = float(lat_vals.mean())
                std_lat   = float(lat_vals.std(ddof=1)) if len(lat_vals) > 1 else 0.0

                per_trial = sub.groupby("trial_id")["is_correct"].mean()
                mean_acc  = float(per_trial.mean())
                std_acc   = float(per_trial.std(ddof=1)) if len(per_trial) > 1 else 0.0

                groups.append({
                    "mean_lat": mean_lat,
                    "std_lat":  std_lat,
                    "accuracy": mean_acc,
                    "std_acc":  std_acc,
                    "n":        len(sub),
                    "color":    color,
                    "marker":   marker,
                    "model":    model_name,
                    "cond":     cond,
                })

        if not groups:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#aaa")
            self._save(fig, "fig04_latency_accuracy_tradeoff.pdf"); return

        for g in groups:
            ax.errorbar(g["mean_lat"], g["accuracy"],
                        xerr=g["std_lat"] if g["std_lat"] > 0 else None,
                        yerr=g["std_acc"] if g["std_acc"] > 0 else None,
                        fmt="none", color=g["color"], lw=0.5,
                        capsize=2, capthick=0.5, alpha=0.6, zorder=3)
            size = max(20, min(80, g["n"] * 7))
            ax.scatter(g["mean_lat"], g["accuracy"],
                       s=size, color=g["color"], marker=g["marker"],
                       alpha=0.88, zorder=5, edgecolors="#333", lw=0.7)

        ax.set_xlabel("Mean Inference Latency (s)", fontsize=8)
        ax.set_ylabel("Prediction Accuracy", fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        max_lat = max(g["mean_lat"] + g["std_lat"] for g in groups)
        ax.set_xlim(0, max_lat * 1.20)

        color_handles = [
            mpatches.Patch(color=PALETTE["local"],  label=self._model_label("local")),
            mpatches.Patch(color=PALETTE["hf_api"], label=self._model_label("hf_api")),
            mpatches.Patch(color=PALETTE["vlm"],    label="Gemini (VLM)"),
        ]
        marker_handles = [
            Line2D([0], [0], marker=mk, lw=0, color="#555",
                   markersize=7, markeredgecolor="#333",
                   label=CONDITION_LABELS[c])
            for c, mk in COND_MARKER.items()
        ]
        ax.legend(handles=color_handles + marker_handles,
                  fontsize=6.2, framealpha=0.9, loc="lower right",
                  title="Model  /  Condition", title_fontsize=6.5)

        ax.set_title("Latency–Accuracy Trade-off  (error bars = ±1 SD)", fontsize=9)
        self._pilot_note(ax)
        fig.tight_layout()
        self._save(fig, "fig04_latency_accuracy_tradeoff.pdf")

    # ── Fig 07 (NEW NUMBER) — LLM / VLM call counts (no scatter) ────────────

    def fig07_call_counts(self):
        """Per-condition bar chart of LLM, VLM, and total calls — no scatter dots."""
        inference_conds = ["B_no_context", "B_with_context",
                           "C_with_filter", "C_with_filter_and_context"]
        series = [
            ("llm_call_count",   PALETTE["llm"],  "LLM calls"),
            ("vlm_call_count",   PALETTE["vlm"],  "VLM calls"),
            ("total_call_count", "#555555",        "Total calls"),
        ]
        n_series = len(series)
        width    = 0.20
        x        = np.arange(len(inference_conds))

        fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.75, 3.6))

        for i, (col, color, label) in enumerate(series):
            offset = (i - (n_series - 1) / 2) * (width + 0.02)
            for xi, cond in enumerate(inference_conds):
                sub   = self.s[self.s["condition"] == cond]
                vals  = sub[col].dropna().values.astype(float)
                mean_v = float(np.nanmean(vals)) if len(vals) else np.nan
                std_v  = float(np.nanstd(vals))  if len(vals) > 1 else 0.0

                ax.bar(x[xi] + offset, mean_v if not np.isnan(mean_v) else 0,
                       width=width, color=color, alpha=0.55,
                       label=label if xi == 0 else "_nolegend_", zorder=3)

                if not np.isnan(mean_v) and std_v > 0:
                    ax.errorbar(x[xi] + offset, mean_v, yerr=std_v,
                                fmt="none", color="#333", lw=0.8,
                                capsize=3, zorder=5)

        ax.set_xticks(list(x))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in inference_conds],
                           fontsize=7.5, rotation=20, ha="right")
        ax.set_ylabel("Calls per trial")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title("LLM / VLM Calls per Trial")
        self._pilot_note(ax)
        fig.tight_layout()
        self._save(fig, "fig06_call_counts.pdf")

    # ── Fig 07 (NEW) — Filter: hint injection outcomes ───────────────────────

    def fig08_filter_effectiveness(self):
        """When the scene-consistency filter activates and injects a hint,
        does the next (augmented) prediction recover to a correct answer?

        Panel A — overall recovery rate (stacked bar: correct / wrong / unclear).
        Panel B — per-assembly-step recovery rate (only steps with ≥1 activation).

        Falls back to a placeholder when no hint-injected calls are present.
        """
        # ── gather hint-injected calls ────────────────────────────────────────
        placeholder_msg = None
        hint_calls = pd.DataFrame()

        if self.p.empty:
            placeholder_msg = "No prediction data available."
        elif "was_hint_injection" not in self.p.columns:
            placeholder_msg = "No hint-injection records in predictions\n(field missing — old log format?)."
        else:
            hint_calls = self.p[
                (self.p["condition"].isin(FILTER_CONDITIONS)) &
                (self.p["was_hint_injection"] == True)
            ].copy()
            if hint_calls.empty:
                placeholder_msg = "No filter activations yet\n(C_with_filter trials pending)."

        if placeholder_msg is not None:
            fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.85, 3.8))
            ax.text(0.5, 0.5, placeholder_msg,
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#aaa", style="italic",
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#ddd"))
            ax.set_title("Scene-Consistency Filter: Hint Injection Outcomes")
            ax.axis("off")
            self._pilot_note(ax)
            fig.tight_layout()
            self._save(fig, "fig07a_filter_effectiveness.pdf")
            return

        # ── classify each hint call ───────────────────────────────────────────
        annotated = hint_calls[hint_calls["is_correct"].notna()].copy()
        n_total   = len(hint_calls)
        n_ann     = len(annotated)
        n_correct = int(annotated["is_correct"].sum()) if not annotated.empty else 0
        n_wrong   = n_ann - n_correct
        n_unclear = n_total - n_ann
        recovery_rate = n_correct / n_ann if n_ann > 0 else float("nan")

        COLOR_CORRECT = "#27ae60"
        COLOR_WRONG   = "#e74c3c"
        COLOR_UNCLEAR = "#aaaaaa"

        fig_a, ax_a = plt.subplots(1, 1, figsize=(DOUBLE_COL * 0.55, 4.0))
        fig_b, ax_b = plt.subplots(1, 1, figsize=(DOUBLE_COL * 0.55, 4.0))

        # ── Panel A: per-trial mean filter activations (stacked) ──────────────
        # Each trial contributes equally; mean ± SD is more representative than
        # a pooled total which inflates high-activation trials.
        trial_ids = sorted(hint_calls["trial_id"].unique())
        n_trials  = len(trial_ids)

        per_trial = []
        for tid in trial_ids:
            sub_t = hint_calls[hint_calls["trial_id"] == tid]
            ann_t = sub_t[sub_t["is_correct"].notna()]
            n_corr_t = int(ann_t["is_correct"].sum()) if not ann_t.empty else 0
            per_trial.append({
                "total":   len(sub_t),
                "correct": n_corr_t,
                "wrong":   len(ann_t) - n_corr_t,
                "unclear": len(sub_t) - len(ann_t),
            })

        totals_arr  = np.array([p["total"]   for p in per_trial], dtype=float)
        corrects_arr = np.array([p["correct"] for p in per_trial], dtype=float)
        wrongs_arr   = np.array([p["wrong"]   for p in per_trial], dtype=float)
        unclears_arr = np.array([p["unclear"] for p in per_trial], dtype=float)

        mean_corr_t = float(corrects_arr.mean())
        mean_wrong_t = float(wrongs_arr.mean())
        mean_uncl_t  = float(unclears_arr.mean())
        mean_total_t = float(totals_arr.mean())
        std_total_t  = float(totals_arr.std(ddof=0)) if n_trials > 1 else 0.0

        bar_segments = [
            (mean_corr_t, COLOR_CORRECT, "Correct (recovered)"),
            (mean_wrong_t, COLOR_WRONG,  "Wrong (not recovered)"),
            (mean_uncl_t,  COLOR_UNCLEAR, "Unclear / unannotated"),
        ]
        left = 0
        for count, color, label in bar_segments:
            if count > 0:
                ax_a.barh([0], [count], left=left, height=0.42,
                          color=color, alpha=0.82, label=label, zorder=3)
                if count >= 0.25:
                    ax_a.text(left + count / 2, 0,
                              f"{count:.1f}", ha="center", va="center",
                              fontsize=8, color="white", fontweight="bold", zorder=4)
                left += count

        # Individual trial total-activation dots as a strip below the bar
        rng_h = np.random.default_rng(42)
        jitter = rng_h.uniform(-0.10, 0.10, size=n_trials)
        ax_a.scatter(totals_arr, jitter - 0.30,
                     s=22, color="#555", zorder=5,
                     edgecolors="white", lw=0.3, alpha=0.85,
                     label="_nolegend_")

        xlim_max = max(mean_total_t + std_total_t + 1.0, mean_total_t * 1.6 + 0.5)
        ax_a.set_xlim(0, xlim_max)
        ax_a.set_ylim(-0.62, 0.60)
        ax_a.set_yticks([])
        ax_a.set_xlabel("Mean filter activations per trial", fontsize=8)
        rate_str = f"{recovery_rate:.0%}" if not np.isnan(recovery_rate) else "—"
        ax_a.set_title(
            f"Recovery Rate: {rate_str}\n"
            f"(mean {mean_total_t:.1f} ± {std_total_t:.1f} activations/trial,"
            f"  N = {n_trials} trials)",
            fontsize=9,
        )
        ax_a.legend(loc="lower right", fontsize=6.5, framealpha=0.9)

        note_a = "Hint = augmented prompt injected after filter rejects a prediction."
        fig_a.text(0.01, -0.02, note_a, fontsize=5.5, color="#888888", ha="left")
        self._pilot_note(ax_a)
        fig_a.tight_layout()
        self._save(fig_a, "fig07a_filter_effectiveness.pdf")

        # ── Panel B: per-step recovery rate ───────────────────────────────────
        # n is shown as share of all annotated filter activations so the reader
        # can see which assembly steps triggered the filter most often.
        step_data = []
        for step in range(1, N_STEPS + 1):
            sub_s = annotated[annotated["gt_step_int"] == step]
            if sub_s.empty:
                continue
            step_data.append({
                "step":    step,
                "n":       len(sub_s),
                "correct": int(sub_s["is_correct"].sum()),
                "rate":    float(sub_s["is_correct"].mean()),
            })

        if step_data:
            xs     = np.arange(len(step_data))
            rates  = [d["rate"] for d in step_data]
            ns     = [d["n"]    for d in step_data]
            colors = [STEP_COLORS[d["step"] - 1] for d in step_data]
            labels = [f"Step {d['step']}" for d in step_data]

            ax_b.bar(xs, rates, color=colors, alpha=0.80, zorder=3)
            for i, (r, n) in enumerate(zip(rates, ns)):
                share = n / n_ann * 100 if n_ann > 0 else 0
                ax_b.text(i, r + 0.03,
                          f"{share:.0f}%\n(n={n})",
                          ha="center", fontsize=6.2, color="#333",
                          linespacing=1.3)

            ax_b.set_xticks(xs)
            ax_b.set_xticklabels(labels, fontsize=7.5)
            ax_b.set_ylim(0, 1.35)
            ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax_b.set_ylabel("Recovery rate (hint → correct)", fontsize=8)
            ax_b.axhline(1.0, color="#ccc", lw=0.6, ls="--", zorder=1)
            ax_b.set_title(
                "Recovery Rate per Assembly Step\n"
                "(% = share of all filter activations at that step)",
                fontsize=9,
            )
        else:
            ax_b.text(0.5, 0.5, "No per-step annotation available.",
                      ha="center", va="center", transform=ax_b.transAxes,
                      fontsize=8, color="#aaa", style="italic")
            ax_b.set_title("Recovery Rate per Assembly Step", fontsize=9)

        fig_b.tight_layout()
        self._save(fig_b, "fig07b_filter_effectiveness.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_reports(metrics: MetricsEngine, output_dir: Path):
    s        = metrics.summaries_df
    p        = metrics.predictions_df
    step_acc = metrics.step_accuracy_df

    # ── metrics_summary + step_accuracy → single .xlsx workbook ─────────────
    # Writing directly to Excel avoids all CSV delimiter / encoding issues;
    # double-clicking the file opens it in Excel with no import wizard.
    xlsx_path = output_dir / "metrics_summary.xlsx"
    sheets = {
        "metrics_summary": s.drop(columns=["_summary_path"], errors="ignore"),
    }
    if not step_acc.empty:
        sheets["step_accuracy"] = step_acc

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"    -> {xlsx_path.name}")
    except ImportError:
        # openpyxl not installed — fall back to semicolon-delimited CSV
        print("    [warn] openpyxl not found; falling back to CSV (sep=;)")
        for name, df in sheets.items():
            fallback = output_dir / f"{name}.csv"
            df.to_csv(fallback, index=False, sep=";", encoding="utf-8-sig")
            print(f"    -> {fallback.name}")

    # ── pilot_report.txt ──────────────────────────────────────────────────────
    W = 64
    lines = [
        "=" * W,
        "  PILOT STUDY REPORT — HRC Assembly System Evaluation",
        "=" * W,
        f"  Trials: {len(s)}   Participants: {s['participant_id'].nunique()}",
        "",
    ]

    _BACKEND_LABEL = {"hf_api": "Llama 4 Scout", "local": "Qwen3 4B INT8"}

    def _model_name(backend: str) -> str:
        return _BACKEND_LABEL.get(backend, backend)

    for cond in CONDITIONS:
        sub = s[s["condition"] == cond]
        lines.append(f"  [{cond}]  N = {len(sub)} trial(s)")
        if sub.empty:
            lines.append("    — no data —\n"); continue

        def _fmt(col, fmt=".1f", suffix=""):
            v = sub[col].dropna()
            if v.empty: return "—"
            m = v.mean(); sd = v.std()
            return f"{m:{fmt}}{suffix}  (σ={sd:{fmt}}{suffix})"

        lines.append(f"    ATCT:            {_fmt('atct_seconds', '.1f', ' s')}")
        lines.append(f"    LLM calls/trial: {_fmt('llm_call_count', '.1f')}")
        lines.append(f"    VLM calls/trial: {_fmt('vlm_call_count', '.1f')}")
        lines.append(f"    Total calls/trial:{_fmt('total_call_count', '.1f')}")
        lines.append(f"    VLM escalation:  {_fmt('vlm_escalation_rate', '.1%')}")
        lines.append(f"    Exec score:      {_fmt('execution_success_score', '.2f')}")

        # LLM accuracy — split by backend (Llama 4 Scout vs Qwen3 local)
        if not p.empty and "llm_backend" in p.columns:
            cond_llm = p[(p["condition"] == cond) & (p["type"] == "llm")]
            backends_present = sorted(cond_llm["llm_backend"].dropna().unique())
            if backends_present:
                for backend in backends_present:
                    sub_b = cond_llm[
                        (cond_llm["llm_backend"] == backend) &
                        cond_llm["is_correct"].notna()
                    ]
                    if sub_b.empty:
                        acc_str = "—"
                    else:
                        acc_str = f"{sub_b['is_correct'].mean():.1%}  (n={len(sub_b)})"
                    label = _model_name(backend)
                    lines.append(f"    LLM accuracy [{label}]:  {acc_str}")
            elif "llm_accuracy" in sub.columns:
                lines.append(f"    LLM accuracy:    {_fmt('llm_accuracy', '.1%')}")
        elif "llm_accuracy" in sub.columns:
            lines.append(f"    LLM accuracy:    {_fmt('llm_accuracy', '.1%')}")

        if "vlm_accuracy" in sub.columns:
            lines.append(f"    VLM accuracy:    {_fmt('vlm_accuracy', '.1%')}")
        lines.append("")

    if not p.empty:
        ann  = p["prediction_correct"].notna() & (p["prediction_correct"] != "unclear")
        corr = p[ann & p["is_correct"].notna()]

        llm_calls = p[p["type"] == "llm"]
        vlm_calls = p[p["type"] == "vlm"]

        lines += [
            "  Per-call summary (all trials combined):",
            f"    Total calls:       {len(p)}",
            f"    LLM calls:         {len(llm_calls)}",
        ]

        # Break LLM calls down by backend
        if "llm_backend" in llm_calls.columns:
            for backend in sorted(llm_calls["llm_backend"].dropna().unique()):
                n_b = (llm_calls["llm_backend"] == backend).sum()
                sub_b = llm_calls[
                    (llm_calls["llm_backend"] == backend) &
                    llm_calls["is_correct"].notna()
                ]
                acc_str = (f"{sub_b['is_correct'].mean():.1%}"
                           if not sub_b.empty else "—")
                label = _model_name(backend)
                lines.append(
                    f"      [{label}]:  {n_b} calls  accuracy={acc_str}"
                )

        lines += [
            f"    VLM calls:         {len(vlm_calls)}",
            f"    Annotated:         {ann.sum()} / {len(p)}",
            f"    Overall accuracy:  "
            + (f"{corr['is_correct'].mean():.1%}" if not corr.empty else "—"),
            "",
        ]

    if not step_acc.empty:
        lines.append("  Per-step accuracy:")
        for step in range(1, N_STEPS + 1):
            def _step_val(ct):
                row = step_acc[(step_acc["step"] == step) & (step_acc["type"] == ct)]
                if row.empty or row["n_calls"].values[0] == 0:
                    return "—"
                return (f"{row['accuracy'].values[0]:.0%}"
                        f"  (n={int(row['n_calls'].values[0])})")
            lines.append(f"    Step {step}:  LLM={_step_val('llm')}  "
                         f"VLM={_step_val('vlm')}")
        lines.append("")

    # ── Filter Analysis block ─────────────────────────────────────────────────
    if "filter_enabled" in s.columns:
        filter_trials = s[s["filter_enabled"] == True]
        if not filter_trials.empty:
            def _fsum(col):
                return int(filter_trials[col].sum()) if col in filter_trials.columns else 0
            lines += [
                "  [Filter Analysis]",
                f"    Trials with filter enabled: {len(filter_trials)}",
                f"    Validation failures: "
                f"LLM={_fsum('validation_failures_llm')}, "
                f"VLM={_fsum('validation_failures_vlm')}, "
                f"Total={_fsum('validation_failures_total')}",
                f"    Hint convergence → valid: "
                f"LLM={_fsum('hint_converged_valid_llm')}, "
                f"VLM={_fsum('hint_converged_valid_vlm')}",
                f"    Hint convergence → none:  "
                f"LLM={_fsum('hint_converged_none_llm')}, "
                f"VLM={_fsum('hint_converged_none_vlm')}",
                f"    Dropped after hint:       "
                f"LLM={_fsum('drops_after_hint_llm')}, "
                f"VLM={_fsum('drops_after_hint_vlm')}",
                "",
            ]

    quality_notes = []
    if s[s["condition"] == "A_no_robot"].empty:
        quality_notes.append("    • Condition A_no_robot: no data — pending collection")
    if not p.empty and "llm_backend" in p.columns:
        if "local" not in p["llm_backend"].dropna().unique():
            quality_notes.append("    • Local LLM backend:   no data — pending trials")
    else:
        quality_notes.append("    • Local LLM backend:   no data — pending trials")
    quality_notes += [
        "    • fetch_time_s:        not yet logged in trial_logger.py",
        "    • execution_success_score: integer inputs auto-normalized ÷ N_STEPS",
    ]
    lines += ["  Data quality notes:"] + quality_notes + ["=" * W]

    report_path = output_dir / "pilot_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"    -> {report_path.name}")

    # ── filter_events sheet in xlsx (when available) ─────────────────────────
    if hasattr(metrics, "filter_events_df") and not metrics.filter_events_df.empty:
        fe_path = output_dir / "filter_events.xlsx"
        try:
            with pd.ExcelWriter(fe_path, engine="openpyxl") as writer:
                metrics.filter_events_df.to_excel(writer, sheet_name="filter_events",
                                                   index=False)
            print(f"    -> {fe_path.name}")
        except ImportError:
            fe_csv = output_dir / "filter_events.csv"
            metrics.filter_events_df.to_csv(fe_csv, index=False, sep=";",
                                             encoding="utf-8-sig")
            print(f"    -> {fe_csv.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# BORIS FLUENCY LOADER
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

class BorisLoader:
    """
    Loads BORIS behavioral annotation CSV exports and computes the
    non-assembly time ratio per observation.

    No mapping file is needed. Videos are renamed by match_videos.py to:
        trial_P001_T05_B_with_context.MOV
    BORIS uses the filename (without extension) as the observation ID.
    Participant, trial tag, and condition are parsed directly from that name.
    The atct_seconds denominator is read from the matching *_summary.json.

    Behavior recognised in BORIS (case-insensitive):
        non_assembly / non assembly / not assembly / wait / fetch / idle …
    Any interval marked with this behavior counts as non-assembly time.

    Metric computed
    ---------------
        not_assembly_ratio = non_assembly_time_s / atct_seconds
        (denominator falls back to BORIS total_time_s for A_no_robot,
         which has no trial logger output)

    BORIS export formats supported (auto-detected)
    -----------------------------------------------
    1. Aggregated events  — Observation id, Behavior, Duration (s), Total length (s)
    2. Tabular events     — Observation id, Time, Behavior, Status (START/STOP)
    """

    # Observation ID pattern: trial_<PID>_T<NN>_<condition>
    _OBS_RE = _re.compile(
        r"^trial_(?P<pid>[^_]+(?:_[^_]+)*)_T(?P<tnum>\d+)_(?P<cond>.+)$",
        _re.IGNORECASE,
    )

    # Video extensions that BORIS appends to observation IDs when loaded from file
    _VIDEO_EXTS = _re.compile(
        r'\.(mov|mp4|avi|mkv|wmv|mts|m4v|mpg|mpeg)$', _re.IGNORECASE
    )

    @classmethod
    def _strip_video_ext(cls, obs_id: str) -> str:
        """Remove trailing video extension from a BORIS observation ID."""
        return cls._VIDEO_EXTS.sub("", obs_id.strip())

    # Single behavior: any interval the operator is NOT assembling.
    # Add here whatever name you use in BORIS (case-insensitive).
    ALIASES: set[str] = {
        "non_assembly", "non assembly", "not assembly", "not_assembly",
        "non-assembly", "idle", "wait", "waiting",
        "fetch", "robot fetch", "robot_fetch",
    }

    def __init__(self, boris_dir: str, log_dir: str):
        self.boris_dir = Path(boris_dir)
        self.log_dir   = Path(log_dir)
        self.fluency_df = self._load_all()

    # ── atct lookup ───────────────────────────────────────────────────────────

    def _atct_from_summary(self, pid: str, trial_tag: str, condition: str) -> float | None:
        """
        Return atct_seconds from the matching *_summary.json, or None if not found.
        Matches on the summary filename pattern: trial_<pid>_<tag>_<condition>_summary.json
        """
        pattern = f"trial_{pid}_{trial_tag}_{condition}_summary.json"
        sf = self.log_dir / pattern
        if sf.exists():
            with open(sf, encoding="utf-8") as f:
                s = json.load(f)
            atct = s.get("atct_seconds")
            return float(atct) if atct is not None else None
        return None

    # ── main loader ───────────────────────────────────────────────────────────

    def _load_all(self) -> pd.DataFrame:
        # Collect all unique observation IDs from every CSV in boris_dir
        obs_ids = self._discover_obs_ids()
        if not obs_ids:
            print(f"  [BORIS] No observations found in {self.boris_dir}")
            return pd.DataFrame()

        rows = []
        for obs_id, csv_path in obs_ids.items():
            parsed = self._parse_obs_id(obs_id)
            if parsed is None:
                print(f"  [BORIS] SKIP '{obs_id}': name does not match "
                      f"trial_<PID>_T<NN>_<condition> pattern")
                continue

            pid, trial_tag, condition = parsed["pid"], parsed["trial_tag"], parsed["condition"]

            metrics = self._compute_metrics(csv_path, obs_id)
            if metrics is None:
                continue

            atct  = self._atct_from_summary(pid, trial_tag, condition)
            denom = atct if atct is not None else metrics["total_time_s"]
            not_asm_s          = metrics["non_assembly_time_s"]
            not_assembly_ratio = not_asm_s / max(denom, 1e-6)

            rows.append({
                "boris_obs_id":       obs_id,
                "condition":          condition,
                "participant_id":     pid,
                "trial_tag":          trial_tag,
                "atct_seconds":       atct,       # None for A_no_robot
                **metrics,
                "not_assembly_ratio": not_assembly_ratio,
            })
            print(f"  [BORIS] {obs_id}  ({condition})  "
                  f"non_assembly={not_asm_s:.1f}s  "
                  f"denom={'atct' if atct else 'boris_total'}={denom:.1f}s  "
                  f"ratio={not_assembly_ratio:.0%}")

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── observation discovery ─────────────────────────────────────────────────

    def _discover_obs_ids(self) -> dict[str, Path]:
        """
        Return {obs_id: csv_path} for every unique observation ID found in
        boris_dir. Each BORIS CSV may contain one or many observations.
        """
        found: dict[str, Path] = {}
        for csv_path in sorted(self.boris_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip", nrows=500)
            except Exception:
                continue
            lower_cols = {c.strip().lower(): c for c in df.columns}
            obs_col = next((lower_cols[k] for k in lower_cols if "observation" in k), None)
            if obs_col is None:
                continue
            for oid in df[obs_col].dropna().astype(str).str.strip().unique():
                if oid and oid not in found:
                    found[oid] = csv_path
        return found

    def _parse_obs_id(self, obs_id: str) -> dict | None:
        m = self._OBS_RE.match(self._strip_video_ext(obs_id))
        if not m:
            return None
        tnum = int(m.group("tnum"))
        return {
            "pid":       m.group("pid"),
            "trial_tag": f"T{tnum:02d}",
            "condition": m.group("cond"),
        }

    # ── format dispatch ───────────────────────────────────────────────────────

    def _compute_metrics(self, csv_path: Path, obs_id: str) -> dict | None:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
        except Exception as e:
            print(f"  [BORIS] ERROR reading {csv_path.name}: {e}")
            return None

        lower_cols = {c.strip().lower(): c for c in df.columns}

        if any("start" in c and "s" in c for c in lower_cols):
            return self._parse_aggregated(df, obs_id, lower_cols)
        elif "status" in lower_cols or "time" in lower_cols:
            return self._parse_events(df, obs_id, lower_cols)
        else:
            print(f"  [BORIS] WARNING: unrecognised format in {csv_path.name}")
            return None

    # ── format parsers ────────────────────────────────────────────────────────

    def _parse_aggregated(self, df: pd.DataFrame, obs_id: str,
                          lower_cols: dict) -> dict:
        """Parse BORIS 'Aggregated events' export."""
        obs_col = next((lower_cols[k] for k in lower_cols
                        if "observation" in k), None)
        if obs_col:
            norm_id = self._strip_video_ext(obs_id)
            df = df[df[obs_col].astype(str).apply(self._strip_video_ext) == norm_id]

        behav_col  = self._pick_col(lower_cols, ["behavior"])
        # Prefer the per-event "Duration (s)" column over "Coding duration" or
        # "Media duration (s)" which also contain the word "duration".
        dur_col    = (lower_cols.get("duration (s)")
                      or self._pick_col(lower_cols, ["duration"]))
        total_col  = self._pick_col(lower_cols, ["total length", "total", "coding duration", "coding", "length"])

        total_time = None
        if total_col:
            v = pd.to_numeric(df[total_col], errors="coerce").dropna()
            total_time = float(v.iloc[0]) if not v.empty else None

        non_assembly = 0.0
        for _, row in df.iterrows():
            canon = self._canonise(str(row.get(behav_col, "")))
            dur   = float(pd.to_numeric(row.get(dur_col, 0), errors="coerce") or 0)
            if canon == "non_assembly":
                non_assembly += dur

        if total_time is None:
            total_time = non_assembly
        return self._build(total_time, non_assembly)

    def _parse_events(self, df: pd.DataFrame, obs_id: str,
                      lower_cols: dict) -> dict:
        """Parse BORIS tabular events (START/STOP) export."""
        obs_col = next((lower_cols[k] for k in lower_cols
                        if "observation" in k), None)
        if obs_col:
            norm_id = self._strip_video_ext(obs_id)
            df = df[df[obs_col].astype(str).apply(self._strip_video_ext) == norm_id].copy()

        time_col   = self._pick_col(lower_cols, ["time"])
        behav_col  = self._pick_col(lower_cols, ["behavior"])
        status_col = self._pick_col(lower_cols, ["status", "type"])
        total_col  = self._pick_col(lower_cols, ["total", "duration"])

        total_time = None
        if total_col:
            v = pd.to_numeric(df[total_col], errors="coerce").dropna()
            total_time = float(v.iloc[0]) if not v.empty else None

        open_start = None
        non_assembly = 0.0
        max_t = 0.0
        for _, row in df.sort_values(time_col).iterrows():
            t      = float(pd.to_numeric(row.get(time_col, 0), errors="coerce") or 0)
            canon  = self._canonise(str(row.get(behav_col, "")))
            status = str(row.get(status_col, "")).strip().upper()
            max_t  = max(max_t, t)
            if canon == "non_assembly":
                if status == "START":
                    open_start = t
                elif status == "STOP" and open_start is not None:
                    non_assembly += t - open_start
                    open_start = None

        if total_time is None:
            total_time = max_t
        return self._build(total_time, non_assembly)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_col(lower_cols: dict, keywords: list[str]):
        for k in lower_cols:
            if any(kw in k for kw in keywords):
                return lower_cols[k]
        return None

    def _canonise(self, behavior: str) -> str:
        return "non_assembly" if behavior.strip().lower() in self.ALIASES else "other"

    @staticmethod
    def _build(total_s: float, non_assembly: float) -> dict:
        return {
            "total_time_s":       total_s,
            "non_assembly_time_s": non_assembly,
        }


# ─────────────────────────────────────────────────────────────────────────────
# BORIS FIGURES  (added to FigureFactory as standalone functions so they
#                 receive the boris_df argument explicitly)
# ─────────────────────────────────────────────────────────────────────────────

def _fig08_fluency_condition(ff: "FigureFactory", boris_df: pd.DataFrame):
    """
    Fig 08 — Non-Assembly Time Ratio: No Robot vs With Robot.

    Compares only A_no_robot and B_with_context (robot assistance with context).
    Metric = non_assembly_time_s (BORIS) / atct_seconds (trial logger).
    Lower is better: the operator spends less time idle / waiting for the robot.
    """
    COMPARE_CONDS  = ["A_no_robot", "B_with_context"]
    COMPARE_LABELS = {"A_no_robot": "No Robot", "B_with_context": "With Robot"}

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, 4.0))
    rng = np.random.default_rng(10)

    if boris_df.empty:
        ax.text(0.5, 0.5,
                "No BORIS data.\nRun with --boris-dir.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#aaa", style="italic",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd"))
        ax.set_title("Non-Assembly Time: No Robot vs With Robot")
        ff._save(fig, "fig08_non_assembly_ratio.pdf"); return

    for xi, cond in enumerate(COMPARE_CONDS):
        color = ff._cond_color(cond)
        sub   = boris_df[boris_df["condition"] == cond]

        if sub.empty:
            ax.text(xi, 0.25, "pending", ha="center", va="center",
                    fontsize=7, color="#bbb", style="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ddd"))
            continue

        vals = sub["not_assembly_ratio"].dropna().values

        jitter = rng.uniform(-0.09, 0.09, size=len(vals))
        xs_d   = [xi + jitter[j] for j in range(len(vals))]

        for j, v in enumerate(vals):
            ax.scatter(xs_d[j], v, s=55, color=color, zorder=5,
                       marker="o", alpha=0.82, edgecolors="white", lw=0.8)

        if len(vals):
            mean_v = float(np.nanmean(vals))
            std_v  = float(np.nanstd(vals)) if len(vals) > 1 else 0.0
            ax.hlines(mean_v, xi - 0.28, xi + 0.28,
                      colors=color, lw=2.2, ls="--", zorder=4)
            ax.text(xi + 0.32, mean_v, f"{mean_v:.0%}",
                    va="center", fontsize=8, color=color, fontweight="bold")
            if std_v > 0:
                ax.errorbar(xi, mean_v, yerr=std_v, fmt="none",
                            color=color, lw=1.2, capsize=5, zorder=3)

    ax.set_xticks(range(len(COMPARE_CONDS)))
    ax.set_xticklabels([COMPARE_LABELS[c] for c in COMPARE_CONDS],
                       fontsize=10)
    ax.set_ylabel("Non-assembly time / ATCT", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(-0.6, len(COMPARE_CONDS) - 0.4)
    ax.set_title("Non-Assembly Time: No Robot vs With Robot\n"
                 "(With Robot = B_with_context trials only)", fontsize=9)

    note = ("Non-assembly time from BORIS annotation / ATCT from trial logger.  "
            "Lower = less idle/wait time.")
    ax.annotate(note, xy=(0.0, -0.16), xycoords="axes fraction",
                fontsize=5.5, color="#888888", ha="left")
    fig.tight_layout()
    ff._save(fig, "fig08_non_assembly_ratio.pdf")


def _fig09_fluency_breakdown(ff: "FigureFactory", boris_df: pd.DataFrame):
    """
    Fig 09 — Non-Assembly Time Ratio per Trial.

    One horizontal bar per observation showing non_assembly_ratio,
    coloured by condition. Sorted by condition then trial tag.
    """
    if boris_df.empty:
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.5))
        ax.text(0.5, 0.5,
                "No BORIS data.\nRun with --boris-dir and --boris-map.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#aaa", style="italic")
        ff._save(fig, "fig09_non_assembly_pertrial.pdf"); return

    df = boris_df.sort_values(["condition", "boris_obs_id"]).reset_index(drop=True)
    n  = len(df)
    fig_h = max(2.5, n * 0.55 + 1.0)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, fig_h))

    y_pos = np.arange(n)
    vals  = df["not_assembly_ratio"].fillna(0).values
    colors = [ff._cond_color(c) for c in df["condition"]]

    ax.barh(y_pos, vals, height=0.6, color=colors, zorder=3)

    for yi, v in enumerate(vals):
        ax.text(v + 0.01, yi, f"{v:.0%}",
                ha="left", va="center", fontsize=7, color="#333")

    y_labels = []
    for _, row in df.iterrows():
        tag  = row.get("trial_tag") or row["boris_obs_id"]
        cond = row["condition"].replace("B_", "").replace("_", " ")
        y_labels.append(f"{tag}  [{cond}]")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=7.5)

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Non-assembly time / ATCT")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axvline(1.0, color="#aaa", lw=0.7, ls="--", zorder=1)

    legend_patches = [
        mpatches.Patch(color=PALETTE[c], label=CONDITION_LABELS[c])
        for c in CONDITIONS if c in PALETTE
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.9)
    ax.set_title("Non-Assembly Time Ratio per Trial  (BORIS / trial logger)")
    ax.invert_yaxis()
    fig.tight_layout()
    ff._save(fig, "fig09_non_assembly_pertrial.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Resolve paths relative to this script's own directory so the script
    # works correctly whether invoked as:
    #   python logs/analyze_results.py      (from project root)
    #   python analyze_results.py           (from inside logs/)
    _HERE = Path(__file__).parent.resolve()

    ap = argparse.ArgumentParser(
        description="Generate publication-quality figures for the HRC evaluation."
    )
    ap.add_argument("--log-dir",    default=str(_HERE),
                    help="Directory with *_summary.json and *_predictions.jsonl "
                         "(default: same directory as this script)")
    ap.add_argument("--output-dir", default=str(_HERE / "analysis"),
                    help="Root output directory; figures go to <output>/figures/ "
                         "(default: logs/analysis/)")
    ap.add_argument("--memory",     default=None,
                    help="Path to memory.json (auto-detected if omitted)")
    ap.add_argument("--boris-dir",  default=str(_HERE / "video"),
                    help="Directory with BORIS CSV exports (default: logs/video/). "
                         "Observation IDs must follow the trial_<PID>_T<NN>_<cond> "
                         "pattern produced by match_videos.py --apply.")
    ap.add_argument("--participants", default=None,
                    help="Comma-separated list of participant IDs to include "
                         "(e.g. P01,P05,P08). Default: all participants.")
    args = ap.parse_args()

    fig_dir = Path(args.output_dir) / "figures"

    bar = "=" * 55
    print(f"\n{bar}")
    print("  HRC Analysis Pipeline")
    print(bar)
    print(f"  Log dir    : {args.log_dir}")
    print(f"  Output dir : {fig_dir}")
    if args.boris_dir:
        print(f"  BORIS dir  : {args.boris_dir}")
    print()

    # ── 1. Load trial logger data ─────────────────────────────────────────────
    print("[1/4]  Loading trial logger data ...")
    loader = DataLoader(args.log_dir, memory_path=args.memory)

    if args.participants:
        pids = [p.strip() for p in args.participants.split(",") if p.strip()]
        print(f"       Filtering to participants: {pids}")
        loader.summaries_df    = loader.summaries_df[
            loader.summaries_df["participant_id"].astype(str).isin(pids)
        ].reset_index(drop=True)
        if not loader.predictions_df.empty:
            loader.predictions_df = loader.predictions_df[
                loader.predictions_df["participant_id"].astype(str).isin(pids)
            ].reset_index(drop=True)
        if not loader.filter_events_df.empty:
            loader.filter_events_df = loader.filter_events_df[
                loader.filter_events_df["participant_id"].astype(str).isin(pids)
            ].reset_index(drop=True)

    print(f"       {len(loader.summaries_df)} trials  |  "
          f"{len(loader.predictions_df)} prediction records\n")

    # ── 2. Compute metrics ────────────────────────────────────────────────────
    print("[2/4]  Computing metrics ...")
    metrics = MetricsEngine(loader.summaries_df, loader.predictions_df,
                            loader.memory, loader.filter_events_df)
    print()

    # ── 3. Load BORIS fluency data (optional) ─────────────────────────────────
    boris_df = pd.DataFrame()
    if args.boris_dir:
        print("[3/4]  Loading BORIS data ...")
        bl = BorisLoader(args.boris_dir, log_dir=args.log_dir)
        boris_df = bl.fluency_df
        print()
    else:
        print("[3/4]  BORIS data skipped  (pass --boris-dir to enable)\n")

    # ── 4. Generate figures ───────────────────────────────────────────────────
    print("[4/4]  Generating figures ...")
    ff = FigureFactory(metrics, fig_dir)

    figs = [
        ("fig01", "Task completion time + execution success",
                  ff.fig01_overview),
        ("fig02", "Bayesian belief evolution (P08 T02)",
                  ff.fig03_bayesian_heatmap),
        ("fig03", "Confusion matrix",
                  ff.fig04_confusion_matrix),
        ("fig04", "Latency–accuracy trade-off with ±SD",
                  ff.fig05_latency_accuracy_tradeoff),
        ("fig05", "Dispatch accuracy vs dispatch rate",
                  ff.fig06_dispatch_quality),
        ("fig06", "LLM / VLM call counts per trial",
                  ff.fig07_call_counts),
        ("fig07", "Filter effectiveness",
                  ff.fig08_filter_effectiveness),
        ("fig08", "Non-assembly time ratio per condition (BORIS)",
                  lambda: _fig08_fluency_condition(ff, boris_df)),
    ]

    ok, failed = 0, []
    for tag, desc, fn in figs:
        print(f"  {tag}  {desc}")
        try:
            fn()
            ok += 1
        except Exception as exc:
            import traceback
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            failed.append(tag)

    print()
    print("  Generating reports ...")
    generate_reports(metrics, Path(args.output_dir))

    print(f"\n{bar}")
    print(f"  Done -- {ok}/{len(figs)} figures generated.")
    if failed:
        print(f"  Failed : {', '.join(failed)}")
    print(f"  Output  : {fig_dir}")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
