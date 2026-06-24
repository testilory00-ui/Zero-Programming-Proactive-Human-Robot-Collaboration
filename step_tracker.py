"""
Step completion tracker for HRC assembly task planning.

Maintains a softmax distribution over assembly steps plus a virtual "completed"
state. Two evidence sources update logits on every call:

  record_prediction(step)       — LLM/VLM predicted step N as current.
                                  Boosts logit[N] by PREDICTION_BOOST and
                                  logit[N+1] by FORWARD_BOOST_RATIO to break
                                  self-reinforcing echoes. Predictions below
                                  the frontier are damped by REGRESSION_DAMPING.

  record_object_visibility(cls) — Robot-camera YOLO scan at HomePose.
                                  Signature-absent steps boost the next step
                                  (VISIBILITY_BOOST); the lowest step still on
                                  the table gets VISIBILITY_PRESENT_BOOST.
                                  No-op when no absence signal exists.

LOGIT_CAP prevents softmax saturation (dominant step peaks at ~97%, not 100%).
P(step j done) = Σ P(current=i) for i > j, sequential monotonicity enforced.
"""

import math
import threading
from collections import Counter

import numpy as np


LOGIT_CAP                = 5.0   # caps softmax; dominant step peaks at ~97%, not 100%
PREDICTION_BOOST         = 1.5   # logit weight per LLM/VLM prediction
VISIBILITY_BOOST         = 2.0   # logit weight per signature-absent step
VISIBILITY_PRESENT_BOOST = 2.0   # logit weight for lowest step still on table; weaker
                                  # because assembled objects can remain camera-visible
DISPATCH_DECAY           = 0.30  # fraction of logit kept for the step being dispatched away from;
                                  # signals it is transitioning to complete (logit ×0.30 ≈ −5 dB)
DISPATCH_BOOST           = 2.0   # logit added to the incoming step on dispatch;
                                  # with forward-boost SS at 2.4, result ≈ 4.8 → ~94% dominant

REGRESSION_GAP     = 0    # predictions this far below frontier are damped;
                           # GAP=0 means any below-frontier prediction is a regression
REGRESSION_DAMPING = 0.15 # boost multiplier on regression; non-zero so repeated
                           # evidence can still override a genuinely wrong frontier

# With decay=0.95, boost=1.5, cap=5.0: fwd=0.12 per call. Steady-state logit
# for step N+1 = 0.12/0.05 = 2.4, well below cap (5.0). Step N dominates
# by exp(5)/exp(2.4) ≈ 13× — no saturation tie, just a gentle nudge forward.
FORWARD_BOOST_RATIO = 0.08

# Words appearing in ≥2 step descriptions — excluded from overlap matching to
# prevent spurious matches on common verbs/prepositions.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "on", "in", "at", "with",
    "for", "by", "onto", "into", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "from",
}


class StepTracker:
    def __init__(self, memory: list[dict], decay_factor: float = 0.95,
                 completion_threshold: float = 0.90):
        self.memory = memory
        self.num_steps = len(memory)
        self.decay_factor = decay_factor
        self.completion_threshold = completion_threshold

        self.logits = self._fresh_logits()
        self.cycle = 1
        self._lock = threading.Lock()
        # Suppress context until first evidence; the startup uniform prior
        # (P(step 1 done)=80%) would mislead the VLM before any action is observed.
        self._has_evidence = False
        # Steps observed as the most-likely current step at least once. A step
        # can only be marked DONE after it has been visited — prevents a single
        # ahead-of-frontier prediction (e.g. LLM guesses step 2 from visible
        # objects while operator is still on step 1) from instantly driving
        # P(step 1 done) past threshold without step 1 ever being "current".
        self._visited_steps: set[int] = set()

        self._signature_objects = self._compute_signature_objects()
        # Words unique to one step — a single match on these is enough to disambiguate.
        self._distinctive_words = self._compute_distinctive_words()

    def _compute_distinctive_words(self) -> dict[int, set[str]]:
        per_step_words: dict[int, set[str]] = {}
        for step in self.memory:
            words = {w for w in step["step description"].lower().split()
                     if w not in _STOPWORDS}
            per_step_words[step["step number"]] = words
        word_freq: Counter = Counter()
        for words in per_step_words.values():
            for w in words:
                word_freq[w] += 1
        return {
            step_num: {w for w in words if word_freq[w] == 1}
            for step_num, words in per_step_words.items()
        }

    def _fresh_logits(self) -> list[float]:
        return [0.0] * (self.num_steps + 1)  # +1 for virtual "completed" slot

    def _update_visited_unlocked(self):
        """Add the current most-likely real step to the visited set. Caller must hold _lock."""
        probs = self._softmax()
        real = probs[: self.num_steps]
        if len(real) == 0:
            return
        idx = int(np.argmax(real))
        self._visited_steps.add(idx + 1)

    # ------------------------------------------------------------------
    # Signature object computation
    # ------------------------------------------------------------------
    def _compute_signature_objects(self) -> dict[int, set[str]]:
        """Step-distinctive objects: required objects minus those shared by
        ≥ ceil(num_steps/2) steps (e.g. screwdriver, carburetor_body)."""
        all_required = [set(s.get("objects_required", [])) for s in self.memory]
        object_freq: Counter = Counter(
            obj for step_objs in all_required for obj in step_objs
        )
        shared_threshold = max(2, math.ceil(self.num_steps / 2))
        shared_tools = {obj for obj, cnt in object_freq.items()
                        if cnt >= shared_threshold}

        sig: dict[int, set[str]] = {}
        for step in self.memory:
            step_num = step["step number"]
            sig[step_num] = set(step.get("objects_required", [])) - shared_tools
        return sig

    @property
    def signature_objects(self) -> dict[int, set[str]]:
        return dict(self._signature_objects)

    # ------------------------------------------------------------------
    # Evidence recording
    # ------------------------------------------------------------------
    def record_prediction(self, predicted_current_step: int):
        """Record an LLM/VLM prediction of the current step.

        Decays all logits, boosts the predicted step (PREDICTION_BOOST) and
        the next step (FORWARD_BOOST_RATIO fraction). The forward boost breaks
        the self-reinforcing lock where ambiguous semantic actions cause the LLM
        to echo the same step indefinitely. Regression predictions (below the
        current frontier) are damped by REGRESSION_DAMPING.
        """
        if predicted_current_step < 1 or predicted_current_step > self.num_steps:
            return
        with self._lock:
            if self._has_evidence:
                frontier = self._frontier_unlocked()
                is_regression = predicted_current_step < (frontier - REGRESSION_GAP)
            else:
                is_regression = False
            boost = PREDICTION_BOOST * (REGRESSION_DAMPING if is_regression else 1.0)

            self._has_evidence = True
            for i in range(self.num_steps + 1):
                self.logits[i] *= self.decay_factor
            idx = predicted_current_step - 1
            self.logits[idx] = min(self.logits[idx] + boost, LOGIT_CAP)

            # Forward boost: step N active → step N+1 approaching.
            # Excluded from virtual completed slot (only visibility absence advances it).
            next_idx = idx + 1
            if next_idx < self.num_steps:
                fwd = boost * FORWARD_BOOST_RATIO
                self.logits[next_idx] = min(self.logits[next_idx] + fwd, LOGIT_CAP)

            self._update_visited_unlocked()
            self._check_cycle_reset()

    def record_object_visibility(self, visible_classes: set[str]):
        """Update logits from a robot-camera YOLO scan at HomePose.

        ABSENCE: steps whose signature objects are all gone boost the next step.
        PRESENCE: the lowest step still visible on table gets a smaller boost.
        Requires at least one absence signal before applying presence — without
        it we cannot distinguish "not yet assembled" from "assembled but still
        camera-visible" (e.g. throttle_stop mounted on the carburetor side).
        No-op if no absence evidence, to avoid flattening the distribution.
        """
        with self._lock:
            absent_next: list[int] = []
            first_present: int | None = None

            for step_num in sorted(self._signature_objects.keys()):
                sig = self._signature_objects[step_num]
                if not sig:
                    continue
                if sig.isdisjoint(visible_classes):
                    next_step = step_num + 1
                    absent_next.append(
                        next_step if next_step <= self.num_steps else self.num_steps + 1
                    )
                elif first_present is None:
                    first_present = step_num

            if not absent_next:
                return

            self._has_evidence = True
            for i in range(self.num_steps + 1):
                self.logits[i] *= self.decay_factor

            for ns in absent_next:
                idx = ns - 1
                self.logits[idx] = min(self.logits[idx] + VISIBILITY_BOOST, LOGIT_CAP)

            if first_present is not None:
                idx = first_present - 1
                self.logits[idx] = min(
                    self.logits[idx] + VISIBILITY_PRESENT_BOOST, LOGIT_CAP
                )

            self._update_visited_unlocked()
            self._check_cycle_reset()

    def record_first_dispatch(self, target_step: int):
        """Called on the FIRST dispatch when no prior step has been completed
        (e.g. VLM stage='idle' at startup). Anchors target_step as the new
        current step. Differs from record_dispatch in that it does NOT decay
        a 'previous' step — there isn't one yet, so we just plant a strong
        boost on target_step and a small forward nudge on target_step+1."""
        if target_step < 1 or target_step > self.num_steps:
            return
        with self._lock:
            self._has_evidence = True
            for i in range(self.num_steps + 1):
                self.logits[i] *= self.decay_factor
            idx = target_step - 1
            self.logits[idx] = min(self.logits[idx] + DISPATCH_BOOST, LOGIT_CAP)
            next_idx = idx + 1
            if next_idx < self.num_steps:
                fwd = DISPATCH_BOOST * FORWARD_BOOST_RATIO
                self.logits[next_idx] = min(self.logits[next_idx] + fwd, LOGIT_CAP)
            self._visited_steps.add(target_step)
            self._update_visited_unlocked()
            self._check_cycle_reset()

    def record_dispatch(self, current_step: int):
        """Called when the robot is dispatched for the objects needed by current_step + 1.

        This is the strongest evidence signal in the system (double-confirmed LLM +
        robot command sent). It marks current_step as transitioning to complete by
        reducing its logit to 30% and gives the following step a head-start boost.

        Effect: next step jumps to ~4.8 logit (≈94% dominant); current step drops
        to ~1.4 logit (≈3% — not zeroed so it can still be corrected if the
        dispatch was wrong); P(current_step done) ≈ 97% but not 100%.
        """
        if current_step < 1 or current_step > self.num_steps:
            return
        with self._lock:
            self._has_evidence = True
            for i in range(self.num_steps + 1):
                self.logits[i] *= self.decay_factor
            curr_idx = current_step - 1
            self.logits[curr_idx] *= DISPATCH_DECAY
            next_idx = current_step  # (current_step + 1) − 1 in 0-indexed
            if next_idx < self.num_steps:
                self.logits[next_idx] = min(
                    self.logits[next_idx] + DISPATCH_BOOST, LOGIT_CAP
                )
            # Dispatching for current_step+1's objects implies current_step has been
            # observed and acted upon — mark it visited so it can transition to DONE.
            self._visited_steps.add(current_step)
            self._update_visited_unlocked()
            self._check_cycle_reset()

    # ------------------------------------------------------------------
    # Softmax & probability derivation
    # ------------------------------------------------------------------
    def _softmax(self) -> np.ndarray:
        """Softmax over current logits. Caller must hold _lock."""
        arr = np.array(self.logits)
        arr = arr - arr.max()
        exps = np.exp(arr)
        total = exps.sum()
        n = self.num_steps + 1
        if total == 0:
            return np.full(n, 1.0 / n)
        return exps / total

    def get_step_probabilities(self) -> dict[int, float]:
        """P(current_step = i) for each real step (excludes virtual completed state)."""
        with self._lock:
            probs = self._softmax()
        return {i + 1: probs[i] for i in range(self.num_steps)}

    def get_completion_probabilities(self) -> dict[int, float]:
        """P(step j done) = Σ P(current=i) for i > j, monotonicity enforced.
        Includes the virtual completed state so P(last step done) > 0 once
        the camera confirms the last signature objects are gone.

        A step that has never been observed as the most-likely current step is
        capped just below completion_threshold so it cannot be marked DONE on
        ahead-of-frontier guesses alone."""
        with self._lock:
            probs = self._softmax()
            visited = set(self._visited_steps)

        raw_completion = {}
        for j in range(self.num_steps):
            step_num = j + 1
            raw_completion[step_num] = sum(
                probs[i] for i in range(j + 1, self.num_steps + 1)
            )

        completion = {}
        prev = 1.0
        unvisited_cap = self.completion_threshold - 1e-3
        for step_num in range(1, self.num_steps + 1):
            val = min(raw_completion[step_num], prev)
            if step_num not in visited:
                val = min(val, unvisited_cap)
            completion[step_num] = val
            prev = val

        return completion

    @property
    def completed_steps(self) -> list[int]:
        """Steps with P(done) above completion_threshold."""
        comp = self.get_completion_probabilities()
        return [s for s in range(1, self.num_steps + 1)
                if comp[s] >= self.completion_threshold]

    @property
    def current_frontier(self) -> int:
        """Lowest step not yet completed."""
        return self._frontier_locked()

    def _frontier_locked(self) -> int:
        comp = self.get_completion_probabilities()
        for s in range(1, self.num_steps + 1):
            if comp[s] < self.completion_threshold:
                return s
        return self.num_steps

    def _frontier_unlocked(self) -> int:
        """Frontier without acquiring _lock (caller must hold it)."""
        probs = self._softmax()
        raw = {}
        for j in range(self.num_steps):
            raw[j + 1] = sum(probs[i] for i in range(j + 1, self.num_steps + 1))
        prev = 1.0
        unvisited_cap = self.completion_threshold - 1e-3
        for s in range(1, self.num_steps + 1):
            val = min(raw[s], prev)
            if s not in self._visited_steps:
                val = min(val, unvisited_cap)
            if val < self.completion_threshold:
                return s
            prev = val
        return self.num_steps

    # ------------------------------------------------------------------
    # Cycle reset
    # ------------------------------------------------------------------
    def _check_cycle_reset(self):
        """Reset for a new cycle when the virtual completed slot reaches threshold.
        Requires a prior visibility boost to that slot — logit saturation on
        early steps alone cannot trigger a reset. Caller must hold _lock."""
        if self.logits[self.num_steps] < VISIBILITY_BOOST * 0.5:
            return
        probs = self._softmax()
        if probs[self.num_steps] >= self.completion_threshold:
            self.cycle += 1
            self.logits = self._fresh_logits()
            self._has_evidence = False
            self._visited_steps.clear()
            print(f"[StepTracker] *** Cycle {self.cycle} — all steps completed, resetting. ***")

    # ------------------------------------------------------------------
    # Step number resolution
    # ------------------------------------------------------------------
    def resolve_step_number(self, step_description: str) -> int | None:
        """Match a step description to a step number via content-word overlap.
        Requires ≥2 shared content words, or 1 word unique to exactly one step."""
        if not step_description:
            return None
        desc_words = {w for w in step_description.strip().lower().split()
                      if w not in _STOPWORDS}
        if not desc_words:
            return None
        best_step = None
        best_score = 0
        best_overlap: set[str] = set()
        for step in self.memory:
            step_words = {w for w in step["step description"].lower().split()
                          if w not in _STOPWORDS}
            overlap_words = desc_words & step_words
            overlap = len(overlap_words)
            if overlap > best_score:
                best_score = overlap
                best_step = step["step number"]
                best_overlap = overlap_words
        if best_score >= 2:
            return best_step
        # Single distinctive word (e.g. "spring", "diaphragm") is enough.
        if best_score == 1 and best_step is not None:
            if best_overlap & self._distinctive_words.get(best_step, set()):
                return best_step
        return None

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------
    def completion_summary_for_prompt(self, style: str = "short") -> str:
        """Return a context string for injection into LLM/VLM prompts.
        style: "short" (default) — one-line directive; "verbose" — per-step listing;
               "none" — empty string."""
        if style == "none":
            return ""
        if not self._has_evidence:
            # Startup default: no evidence yet, but Step 1 is the natural starting
            # point. Without this hint the VLM can guess any step from incidental
            # visible objects (e.g. float_bowl → step 5).
            return (
                "STEP TRACKER: assembly has just begun — no progress evidence yet. "
                "Step 1 is the most likely current step unless the scene clearly "
                "indicates otherwise."
            )
        if style == "verbose":
            return self._verbose_listing()
        return self._short_directive()

    def _short_directive(self) -> str:
        """One-line prior: top step(s) as most-likely current, completed steps listed.
        When the top-2 are within AMBIGUITY_MARGIN, both are named to prevent the
        LLM from anchoring on the nominal winner when the prior is genuinely uncertain."""
        AMBIGUITY_MARGIN = 0.15

        step_probs = self.get_step_probabilities()
        comp_probs = self.get_completion_probabilities()

        sorted_steps = sorted(step_probs, key=step_probs.get, reverse=True)
        best, conf = sorted_steps[0], step_probs[sorted_steps[0]]
        second = sorted_steps[1] if len(sorted_steps) >= 2 else None

        if second is not None and (conf - step_probs[second]) < AMBIGUITY_MARGIN:
            conf2 = step_probs[second]
            earlier, later = min(best, second), max(best, second)
            # Sequential tie (N, N+1) = forward-boost deadlock pattern: Step N has
            # been echoed enough that FORWARD_BOOST_RATIO lifted N+1 to match it.
            # Operator has likely progressed — prefer the later step on ambiguity.
            if later == earlier + 1:
                step_hint = (
                    f"Step {earlier} or Step {later} ({step_probs[earlier]:.0%} vs "
                    f"{step_probs[later]:.0%}) — Step {earlier} has been predicted "
                    f"repeatedly, so the operator has likely progressed to Step {later}. "
                    f"If the observed action could match either, prefer Step {later}"
                )
            else:
                step_hint = (
                    f"Step {best} ({conf:.0%}) or Step {second} ({conf2:.0%}) — "
                    f"ambiguous, use observed action and recent changes to decide"
                )
        else:
            step_hint = f"Step {best} (confidence {conf:.0%})"

        completed = [s for s in range(1, self.num_steps + 1)
                     if comp_probs[s] >= self.completion_threshold]
        completed_str = (
            f" Steps {', '.join(str(s) for s in completed)} appear already done."
            if completed else ""
        )

        return (
            f"STEP TRACKER: the assembly is most likely at {step_hint}.{completed_str} "
            f"This estimate comes from accumulated tracking evidence — lean on it when "
            f"the scene is ambiguous or multiple steps seem plausible, but trust clear "
            f"visual or semantic evidence if it points elsewhere."
        )

    def _verbose_listing(self) -> str:
        """Per-step probability listing. Kept for A/B comparison."""
        step_probs = self.get_step_probabilities()
        comp_probs = self.get_completion_probabilities()

        lines = ["STEP COMPLETION PROBABILITIES (estimated, may have errors):"]
        for s in range(1, self.num_steps + 1):
            lines.append(
                f"- Step {s}: {step_probs[s]:.0%} chance current, "
                f"{comp_probs[s]:.0%} chance already done"
            )
        lines.append(
            "\nUse these as soft hints only. Prioritize what you observe from the scene, "
            "use this to confirm your prediction or helping you to select between two possible candidates. "
        )
        return "\n".join(lines)
