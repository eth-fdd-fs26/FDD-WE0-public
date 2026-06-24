"""Activate a broken-solution scenario so you can watch the panel react to it.

    uv run --project workshop python workshop/scenario.py list
    uv run --project workshop python workshop/scenario.py use p2_wrong
    uv run --project workshop python workshop/scenario.py reset

`use` copies the scenario's solution file(s) into workshop/solutions/ (the same
folder a student drops their work into). Then (re)launch the panel — or just hit
↻ Restart — and watch where it melts down. `reset` clears them again so every
part falls back to the reference solution (the plant survives).
"""
import glob
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from workshop.tests._scenarios import BASELINE, SCENARIOS, scenario_dir  # noqa: E402

SOLUTIONS = os.path.join(_HERE, "solutions")


def _clear_solutions():
    removed = 0
    for f in glob.glob(os.path.join(SOLUTIONS, "solution_part*.py")):
        os.remove(f)
        removed += 1
    return removed


def cmd_list():
    print("\nFailure-mode scenarios (part × kind):\n")
    width = max(len(s["name"]) for s in SCENARIOS)
    for sc in SCENARIOS:
        print(f"  {sc['name']:<{width}}  part {sc['part']} · {sc['kind']:<7} — {sc['blurb']}")
        print(f"  {'':<{width}}  → expect a meltdown at '{sc['step']}' ({sc['title']})")
    print(f"\n  {'reset':<{width}}  remove all drop-ins — every part uses the reference ({BASELINE['blurb']})\n")
    print("Usage:  python workshop/scenario.py use <name>   |   python workshop/scenario.py reset\n")


def cmd_use(name):
    sc = next((s for s in SCENARIOS if s["name"] == name), None)
    if sc is None:
        print(f"Unknown scenario '{name}'. Run  scenario.py list  to see them all.")
        return 1
    _clear_solutions()
    src = scenario_dir(name)
    copied = []
    for f in glob.glob(os.path.join(src, "solution_part*.py")):
        shutil.copy(f, os.path.join(SOLUTIONS, os.path.basename(f)))
        copied.append(os.path.basename(f))
    print(f"\n✅ Activated '{name}'  ({sc['kind']} code in part {sc['part']})")
    print(f"   copied {', '.join(copied)} → workshop/solutions/")
    print(f"   what it is: {sc['blurb']}")
    print(f"   expect:     meltdown at '{sc['step']}' — \"{sc['title']}\"")
    print("\n   Now launch (or hit ↻ Restart):  uv run --project workshop python workshop/launch.py\n")
    return 0


def cmd_reset():
    n = _clear_solutions()
    print(f"\n🧹 Cleared {n} drop-in solution file(s). Every part now uses the reference "
          "solution — the plant should survive to dawn.\n")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        cmd_list(); return 0
    cmd = argv[0]
    if cmd == "list":
        cmd_list(); return 0
    if cmd == "reset":
        return cmd_reset()
    if cmd == "use":
        if len(argv) < 2:
            print("Usage: scenario.py use <name>"); return 1
        return cmd_use(argv[1])
    print(f"Unknown command '{cmd}'. Try: list | use <name> | reset")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
