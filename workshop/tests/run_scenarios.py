"""Headless failure-mode suite.

Runs the whole day for each broken-solution scenario and checks the plant melts
down at the expected step — plus a baseline that should reach victory. No web
layer, no extra dependencies.

    uv run --project workshop python workshop/tests/run_scenarios.py
"""
import asyncio
import os
import sys

# Each scenario folder ships only the ONE part under test; the others must fall
# back to the reference. Enable fallback so strict launch-mode doesn't melt the
# plant down at part 1 before we reach the part we're testing.
os.environ.setdefault("NCM_ALLOW_FALLBACK", "1")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from workshop.backend import loader  # noqa: E402
from workshop.backend.engine import collect_events  # noqa: E402
from workshop.tests._scenarios import BASELINE, SCEN_DIR, SCENARIOS, scenario_dir  # noqa: E402


def _run(solutions_dir):
    loader.SOLUTIONS_DIR = solutions_dir          # point the loader at this scenario
    events = asyncio.run(collect_events(pace=0.0))
    exploded = next((e for e in events if e.status == "exploded"), None)
    final = events[-1].step_id
    return events, exploded, final


def main() -> int:
    cases = [BASELINE] + SCENARIOS
    rows, ok_count = [], 0
    print(f"\nRunning {len(cases)} scenarios through the day engine…\n")
    for sc in cases:
        sdir = SCEN_DIR if sc["kind"] == "baseline" else scenario_dir(sc["name"])
        _events, exploded, final = _run(sdir)

        if sc["kind"] == "baseline":
            passed = final == "day.victory" and exploded is None
            got = f"reached {final}"
            want = "reach day.victory"
        else:
            passed = (exploded is not None
                      and exploded.step_id == sc["step"]
                      and sc["title"].lower() in exploded.title.lower())
            got = f"explode at {exploded.step_id} ({exploded.title})" if exploded else f"no explosion (final {final})"
            want = f"explode at {sc['step']} (~'{sc['title']}')"

        ok_count += passed
        rows.append((sc["name"], sc["kind"], "PASS" if passed else "FAIL", want, got))

    w = max(len(r[0]) for r in rows)
    for name, kind, verdict, want, got in rows:
        mark = "✅" if verdict == "PASS" else "❌"
        print(f"{mark} {name:<{w}}  [{kind:<8}]  expected: {want}")
        print(f"   {'':<{w}}   actual:   {got}")
    print(f"\n{ok_count}/{len(cases)} scenarios behaved as expected.\n")
    return 0 if ok_count == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
