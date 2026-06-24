"""
LLM inference tests targeting two fixes implemented after the pilot study:

  Fix 1 — Forward boost (FORWARD_BOOST_RATIO=0.35 in step_tracker.py)
           Each LLM prediction of step N also boosts step N+1 by 35%.
           After ~8 consecutive step-2 echoes, step-3 logit reaches the cap
           and the prior becomes competitive → breaks the self-reinforcing
           deadlock observed in T11 calls 09-11 (s2=68-74%, GT=step 3).

  Fix 2 — Sticky delta (consume_delta() / _pending_delta_parts in perception.py)
           Object-entry events now persist in _pending_delta_parts until
           consume_delta() is called after LLM dispatch — not cleared by the
           1500ms INVENTORY_DELTA_WINDOW that expires before inference (1–6s).
           The LLM receives "assembly with ... | recent: spring entered" even
           when the snapshot window has already rolled over.

Test matrix (4 cases):
  T1 — Forward boost breaks the deadlock: 9 step-2 echoes while GT=step 3,
       spring occluded (ambiguous action). Tracker should show step 3 ≥40%.
  T2 — Sticky delta alone: flat tracker (robot just joined), "spring entered"
       delta disambiguates step 3 without any tracker history.
  T3 — Both fixes combined: canonical T11 failure scenario. 8 step-2 echoes +
       "spring entered" delta. Both fixes must cooperate to resolve step 3.
  T4 — Regression damping + forward boost coexistence: steps 1–4 done,
       5 spurious step-2 predictions (3 steps below frontier=5). Damping must
       not push the posterior backward; float_bowl identifies step 5.

Assembly steps (learned_memory.json):
  Step 1: throttle_stop, carburetor_body, screwdriver, screws
  Step 2: diaphragm, carburetor_body
  Step 3: spring, carburetor_body
  Step 4: cover, carburetor_body, screwdriver, screws
  Step 5: float_bowl, carburetor_body, screwdriver, screws

Run:
    python other/test_llm.py --backend local    # Local Qwen3 INT4 (OpenVINO)
    python other/test_llm.py --backend hf       # HF Inference API
    python other/test_llm.py --backend gemma    # Gemini API (LLM path)
    python other/test_llm.py                    # all backends

Requires GEMINI_API_KEY / HUGGING_FACE_HUB_TOKEN for non-local backends.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from step_tracker import StepTracker
from llm import LLM_planner


BACKENDS = {
    "local": "infer_LLM",
    "gemma": "infer_LLM_GEMMA",
    "hf":    "infer_LLM_HF",
}
LOCAL_BACKENDS = {"local"}


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
#
# completed_steps   — steps correctly completed before this call (via simulation).
# false_predictions — [(step_num, count)] injected after completion to model
#                     a bad-prior or deadlock scenario.
# ─────────────────────────────────────────────────────────────────────────────
TEST_CASES = [

    # ═════════════════════════════════════════════════════════════════════════
    # T1 — Forward boost breaks the step-2/step-3 deadlock
    #
    # Replays T11 calls 09–11. Step 1 correctly completed; then 9 consecutive
    # step-2 echoes while the operator is already on step 3 (spring occluded
    # by hand, so YOLO only sees diaphragm + carburetor_body).
    #
    # Without Fix 1: logit[step3]=0 throughout → prior says "S2 certain" →
    #   LLM echoes step 2 indefinitely.
    # With Fix 1: each step-2 echo adds 0.7 to logit[step3]. After 9 calls
    #   logit[step3] ≈ 4.4 (cap 4.5), matching logit[step2] → prior shows
    #   step 2 and step 3 as ~50-50 → next LLM call has a fair chance of
    #   escaping to step 3 even without the spring signal.
    # ═════════════════════════════════════════════════════════════════════════
    {
        "group": "T",
        "name": "T1 — Forward boost: deadlock breaks after 9 step-2 echoes (T11 replay)",
        "description": (
            "Step 1 done. 9 consecutive step-2 predictions while GT=step 3 "
            "(spring occluded → semantic action lexically matches step 2). "
            "Fix 1 (forward boost 0.35): each echo boosts step-3 logit by 0.7; "
            "after 9 calls step 3 reaches logit ≈4.4 — context should show step 3 "
            "competitive (~50%) instead of stuck at ~0%."
        ),
        "expected_step": 3,
        "semantic_action": "assembly with carburetor_body, diaphragm",
        "completed_steps": [1],
        "false_predictions": [(2, 9)],
    },

    # ═════════════════════════════════════════════════════════════════════════
    # T2 — Sticky delta alone disambiguates step 3, no tracker history
    #
    # Robot comes online mid-assembly (operator is already on step 3). Tracker
    # is flat — no prior evidence so context is suppressed. The semantic action
    # "assembly with diaphragm, carburetor_body" is lexically identical to step 2.
    # Only the sticky delta "spring entered" distinguishes step 3.
    #
    # Fix 2: _pending_delta_parts persists "spring entered" until consume_delta()
    # is called post-dispatch. Without Fix 2 the 1500ms window would clear the
    # event before inference completes, leaving the LLM with the ambiguous action.
    # ═════════════════════════════════════════════════════════════════════════
    {
        "group": "T",
        "name": "T2 — Sticky delta alone: 'spring entered' resolves step 3 with flat prior",
        "description": (
            "No tracker history (robot just joined). Context is suppressed. "
            "Semantic action 'assembly with diaphragm, carburetor_body' matches "
            "both step 2 and step 3. Fix 2 (sticky delta) keeps 'spring entered' "
            "alive across the 1–6s inference window so the LLM receives the hint."
        ),
        "expected_step": 3,
        "semantic_action": "assembly with diaphragm, carburetor_body | recent: spring entered",
        "completed_steps": [],
        "false_predictions": [],
    },

    # ═════════════════════════════════════════════════════════════════════════
    # T3 — Both fixes combined: canonical T11 failure scenario
    #
    # Step 1 done; 8 step-2 echoes have built up (deadlock period); AND the
    # delta "spring entered" is present (sticky delta kept it alive).
    # Fix 1 makes the prior competitive (~50-50 step 2 vs step 3).
    # Fix 2 delivers the spring signal to the LLM despite inference latency.
    # Together they must resolve step 3 reliably.
    # ═════════════════════════════════════════════════════════════════════════
    {
        "group": "T",
        "name": "T3 — Both fixes: deadlock (8 echoes) + spring delta → step 3",
        "description": (
            "Step 1 done; 8 step-2 echoes have accumulated (forward boost has "
            "built logit[step3]≈4.2, competitive with logit[step2]=4.5). "
            "Sticky delta adds 'spring entered'. Both fixes cooperate: prior "
            "is ambiguous (~50-50) but delta provides the tiebreaker for step 3."
        ),
        "expected_step": 3,
        "semantic_action": "assembly with diaphragm, carburetor_body | recent: spring entered",
        "completed_steps": [1],
        "false_predictions": [(2, 8)],
    },

    # ═════════════════════════════════════════════════════════════════════════
    # T4 — Regression damping does not suppress forward boost (late assembly)
    #
    # Steps 1–4 correctly completed (frontier=5). Then 5 spurious step-2
    # predictions land 3 steps below the frontier → REGRESSION_DAMPING=0.3
    # reduces each boost to 0.6 (instead of 2.0). This must NOT pull the
    # posterior back below step 4. The forward boost from those damped echoes
    # also propagates (damped) to step 3 — correctly low, not interfering.
    # float_bowl (unique to step 5) in the semantic action should confirm step 5.
    # ═════════════════════════════════════════════════════════════════════════
    {
        "group": "T",
        "name": "T4 — Regression damping: 5 step-2 echoes at frontier=5 don't pull backward",
        "description": (
            "Steps 1–4 done (frontier=5). 5 spurious step-2 predictions 3 steps "
            "below frontier — each is damped to 0.3× (REGRESSION_DAMPING). "
            "Damping must keep step 2 far below threshold (~0% done undone). "
            "float_bowl (unique to step 5) confirms the correct step regardless."
        ),
        "expected_step": 5,
        "semantic_action": "assembly with carburetor_body, float_bowl, screwdriver, screws",
        "completed_steps": [1, 2, 3, 4],
        "false_predictions": [(2, 5)],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Tracker simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_step_completion(tracker, completed_step, n_current=3, n_next=5):
    """Simulate main.py's LLM feedback after an operator completes a step.

    n_current predictions at the completed step + n_next at the following step.
    n_next=5 with decay=0.95 gives P(done)≈93%, above completion_threshold=0.90.
    """
    next_step = (completed_step % tracker.num_steps) + 1
    for _ in range(n_current):
        tracker.record_prediction(completed_step)
    for _ in range(n_next):
        tracker.record_prediction(next_step)


def _build_tracker_context(memory, completed_steps, false_predictions,
                           context_style="short"):
    """Return (tracker, context_string) for the given assembly state."""
    tracker = StepTracker(memory)
    for s in completed_steps:
        _simulate_step_completion(tracker, s)
    for fp_step, n in false_predictions:
        for _ in range(n):
            tracker.record_prediction(fp_step)
    return tracker, tracker.completion_summary_for_prompt(style=context_style)


def _context_summary(tracker):
    """Compact per-step summary: S1:XX%done S2:XX%done ..."""
    comp = tracker.get_completion_probabilities()
    probs = tracker.get_step_probabilities()
    parts = [f"S{s}:{probs[s]:.0%}curr/{comp[s]:.0%}done"
             for s in range(1, tracker.num_steps + 1)]
    return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy check
# ─────────────────────────────────────────────────────────────────────────────

def _best_matching_step(got_objs, memory_by_step):
    """Step whose objects_required best overlaps with got_objs (Jaccard)."""
    best_step, best_score = None, -1
    for num, step in memory_by_step.items():
        expected = set(step["objects_required"])
        if not expected:
            continue
        score = len(got_objs & expected) / len(got_objs | expected)
        if score > best_score:
            best_score, best_step = score, num
    return best_step, best_score


def _check_accuracy(parsed, test, memory_by_step):
    """Return (correct: bool, detail: str).

    Correct when 'objects required' in the response best matches the next step
    after expected_step (LLM correctly identified current step and derived next).
    """
    current_step = test["expected_step"]
    total = max(memory_by_step)
    expected_next = (current_step % total) + 1

    got_objs = set(parsed.get("objects required") or [])
    predicted_step, score = _best_matching_step(got_objs, memory_by_step)
    correct = predicted_step == expected_next
    detail = (
        f"expected next=step {expected_next} "
        f"{sorted(memory_by_step[expected_next]['objects_required'])}"
        f"  |  got {sorted(got_objs)}"
        f" → best match step {predicted_step} (score {score:.2f})"
    )
    return correct, detail


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(backend=None, context_style="short"):
    selected = {backend: BACKENDS[backend]} if backend else BACKENDS
    need_local = bool(selected.keys() & LOCAL_BACKENDS)

    print("Loading LLM_planner" + (" with local model..." if need_local else " (API-only)..."))
    planner = LLM_planner(load_local_model=need_local)

    with open("learned_memory.json", "r", encoding="utf-8") as f:
        planner.memory = json.load(f)
    planner.assembly_str = json.dumps(planner.memory, indent=2)
    memory_by_step = {s["step number"]: s for s in planner.memory}

    all_results = {}

    for label, method_name in selected.items():
        print(f"\n{'═' * 72}")
        print(f"  BACKEND : {label}  —  {len(TEST_CASES)} tests  [context: {context_style}]")
        print(f"{'═' * 72}")

        results = []
        for idx, test in enumerate(TEST_CASES, 1):
            tracker, context = _build_tracker_context(
                planner.memory,
                test["completed_steps"],
                test.get("false_predictions", []),
                context_style=context_style,
            )

            scene_data = {"semantic_action": test["semantic_action"]}
            if context_style != "none" and context:
                scene_data["step_completion_context"] = context

            print(f"\n  [{idx:02d}/{len(TEST_CASES)}] {test['name']}")
            print(f"  {'─' * 68}")
            print(f"  desc     : {test['description']}")
            print(f"  action   : {test['semantic_action']}")
            fp = test.get("false_predictions", [])
            if fp:
                print(f"  fp noise : {', '.join(f'S{s}×{n}' for s, n in fp)}")
            print(f"  tracker  : {_context_summary(tracker)}")
            if context_style == "none":
                print(f"  context  : (suppressed)")
            elif context:
                print(f"  context  :")
                for line in context.splitlines():
                    print(f"             {line}")
            else:
                print(f"  context  : (empty — no evidence yet)")

            correct = None
            elapsed = 0.0
            status = "ERROR"
            t0 = time.time()
            try:
                raw = getattr(planner, method_name)(scene_data)
                elapsed = time.time() - t0
                parsed = json.loads(raw)
                status = "OK"
                correct, detail = _check_accuracy(parsed, test, memory_by_step)
                tag = "OK CORRECT" if correct else "!! WRONG  "
                print(f"  elapsed  : {elapsed:.2f}s")
                print(f"  stage    : {parsed.get('stage of assembly')}")
                print(f"  next_op  : {parsed.get('next operation')}")
                print(f"  objects  : {parsed.get('objects required')}")
                print(f"  result   : {tag}  {detail}")
            except json.JSONDecodeError as e:
                elapsed = time.time() - t0
                status = "PARSE ERR"
                print(f"  elapsed  : {elapsed:.2f}s")
                print(f"  result   : ✗ PARSE ERROR  raw={raw!r}  err={e}")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  elapsed  : {elapsed:.2f}s")
                print(f"  result   : ✗ ERROR  {e}")

            results.append({
                "name":    test["name"],
                "status":  status,
                "elapsed": elapsed,
                "correct": correct,
            })

        all_results[label] = results

    print(f"\n{'═' * 72}")
    print("  SUMMARY")
    print(f"{'═' * 72}")
    for label, results in all_results.items():
        ok      = sum(1 for r in results if r["status"] == "OK")
        correct = sum(1 for r in results if r["correct"] is True)
        avg     = sum(r["elapsed"] for r in results) / len(results) if results else 0
        print(f"\n  {label}:  {correct}/{len(results)} correct  "
              f"({ok}/{len(results)} parsed OK)  avg {avg:.2f}s/call")
        for r in results:
            acc = "OK" if r["correct"] is True else ("!!" if r["correct"] is False else "?")
            print(f"    [{r['status']:9s}] [{acc}]  {r['elapsed']:.2f}s  {r['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test LLM inference for the two post-pilot fixes (forward boost + sticky delta)."
    )
    parser.add_argument(
        "--backend", choices=list(BACKENDS.keys()), default=None,
        help="Backend to test. Omit to run all.",
    )
    parser.add_argument(
        "--context-style", choices=["short", "verbose", "none"], default="short",
        help="Prior format injected into scene_data.",
    )
    args = parser.parse_args()
    run_tests(args.backend, context_style=args.context_style)
