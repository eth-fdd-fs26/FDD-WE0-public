# Failure-mode test suite

Does the panel actually react correctly when a student's solution is **missing**,
**wrong**, or just produces **bad results**? This suite covers all three, for all
three parts (a 3×3 matrix), plus an "all correct" baseline.

| | missing code | wrong code | bad results |
|---|---|---|---|
| **Part 1 — pump** | `predict()` not implemented | `predict()` returns probabilities, not 0/1 | `predict()` always says "safe" → misses failures |
| **Part 2 — basin** | regression head missing | layer size flipped → forward crashes | architecture fine but untrained → huge MAE |
| **Part 3 — waste** | `fill_gaps()` not implemented | `route()` re-clusters instead of reusing the fit | `fill_gaps()` fills with a flat mean → R² collapses |

Each scenario is one drop-in file under `scenarios/<name>/solution_part{N}.py`.

## 1. Try it in the interface

Activate a scenario, then launch (or hit ↻ Restart) and watch where it melts down:

```bash
uv run --project workshop python workshop/scenario.py list        # show all scenarios
uv run --project workshop python workshop/scenario.py use p2_wrong
uv run --project workshop python workshop/launch.py               # watch it explode at basin.arch
uv run --project workshop python workshop/scenario.py reset        # back to all-reference → victory
```

`use` copies the scenario's file into `workshop/solutions/` (exactly where a
student drops their work); `reset` clears it again.

## 2. Run the whole suite headlessly

Checks every scenario explodes at the expected step (and the baseline reaches
victory) — no browser, no extra dependencies:

```bash
uv run --project workshop python workshop/tests/run_scenarios.py
```

It prints a PASS/FAIL table and exits non-zero if anything misbehaves.

## How it works

`run_scenarios.py` points `loader.SOLUTIONS_DIR` at each scenario folder and runs
`DayEngine.run()` headlessly, then asserts the first `exploded` event matches the
expected `step`/`title` from `_scenarios.py`. The interactive `scenario.py`
instead copies files into the real `workshop/solutions/` so the running app picks
them up. Scenario metadata (expected outcomes, blurbs) lives in `_scenarios.py`.
