# Drop your solutions here

When you finish a homework notebook, it lets you **export a solution file**.
Drop that file into this folder and the control panel will run *your* code
instead of the built-in reference solution.

| Part | File you drop here       | What it must expose (`get_solution()` returns…)                          |
|------|--------------------------|--------------------------------------------------------------------------|
| 1    | `solution_part1.py`      | `.clean(df) -> df` and `.predict(df) -> 0/1 array`                        |
| 2    | `solution_part2.py`      | `.model` (your trained `nn.Module`) and `.predict_temperature(X) -> °C`  |
| 3    | `solution_part3.py`      | `.fill_gaps(history, gappy) -> df`, `.build_machine_plan(df) -> dict`, `.route(plan, new_df) -> ids` |

Every file must define a top-level factory:

```python
def get_solution():
    return MySolution()   # an object implementing the part's methods above
```

Save any trained artifacts (e.g. `pump_model.joblib`, `basin_net.pt`) **next to
the `.py` file in this folder** and load them inside `get_solution()`.

Anything you drop here is **git-ignored** — it stays on your machine. The
control panel auto-detects it on the next launch; a panel badge shows whether it
ran the `reference` or your `student` solution.

> Need the exact shapes? See `workshop/backend/contracts.py`, and the bundled
> reference answers in `workshop/backend/reference/` are complete working
> examples you can imitate.
