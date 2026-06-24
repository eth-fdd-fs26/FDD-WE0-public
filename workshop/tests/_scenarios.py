"""Catalogue of broken-solution scenarios for the failure-mode test suite.

Each scenario is a folder under ``scenarios/<name>/`` containing a single
``solution_part{N}.py`` that breaks Part N in one of three ways:

  * **missing** — a required piece is not implemented (blank/absent method).
  * **wrong**   — the code runs but does the wrong thing (a real bug).
  * **bad**     — the code is plausible and runs, but the model is too weak.

``step``/``title`` are what the engine is expected to explode on, so both the
interactive activator (``scenario.py``) and the headless suite
(``run_scenarios.py``) can describe / assert the outcome.
"""
import os

SCEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")

SCENARIOS = [
    # --- Part 1: coolant pump -------------------------------------------------
    {"name": "p1_missing", "part": 1, "kind": "missing", "step": "pump.predict",
     "title": "Prediction crashed",
     "blurb": "predict() left unimplemented — the method is simply absent."},
    {"name": "p1_wrong", "part": 1, "kind": "wrong", "step": "pump.predict",
     "title": "Bad prediction output",
     "blurb": "predict() returns probabilities (floats) instead of 0/1 labels."},
    {"name": "p1_bad", "part": 1, "kind": "bad", "step": "pump.predict",
     "title": "CORE OVERHEAT",
     "blurb": "predict() runs but always says 'safe' — misses every real failure."},

    # --- Part 2: basin temperature -------------------------------------------
    {"name": "p2_missing", "part": 2, "kind": "missing", "step": "basin.arch",
     "title": "No regression head",
     "blurb": "the regression head is an Identity — the net outputs 64 numbers, not 1."},
    {"name": "p2_wrong", "part": 2, "kind": "wrong", "step": "basin.arch",
     "title": "Network won't run",
     "blurb": "the head expects 32 but the fusion is 64 — forward() throws a shape error."},
    {"name": "p2_bad", "part": 2, "kind": "bad", "step": "basin.wiring",
     "title": "Vision encoder isn't frozen",
     "blurb": "shapes are fine but the do-not-touch encoder was left trainable."},

    # --- Part 3: waste machines ----------------------------------------------
    {"name": "p3_missing", "part": 3, "kind": "missing", "step": "waste.repair",
     "title": "Gaps left in the log",
     "blurb": "fill_gaps() left unimplemented — the missing values are never filled."},
    {"name": "p3_wrong", "part": 3, "kind": "wrong", "step": "waste.route",
     "title": "mis-routed",
     "blurb": "route() re-clusters the arrivals instead of reusing the fitted machines."},
    {"name": "p3_bad", "part": 3, "kind": "bad", "step": "waste.repair",
     "title": "filled with garbage",
     "blurb": "fill_gaps() runs but fills quantities with a flat mean — R² collapses."},
]

BASELINE = {"name": "_all_correct", "part": 0, "kind": "baseline", "step": "day.victory",
            "title": "", "blurb": "no broken files — every part uses the reference solution."}


def scenario_dir(name: str) -> str:
    return os.path.join(SCEN_DIR, name)
