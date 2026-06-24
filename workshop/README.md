# ☢️ Nuclear Central Manager — the core project

A funny, gamified **nuclear power-plant control panel** that stitches together
the three homework blocks of the course. You've been appointed plant manager;
your job is to **keep the reactor alive for one full day**. Three subsystems
each guard a different failure mode, and each is powered by one of your homework
solutions:

| Subsystem | Homework block | What your code does |
|-----------|----------------|---------------------|
| 🛠️ **Coolant pump** | Part 1 — data cleaning + classifier | Clean the messy sensor logs and predict pump failure before the core overheats. |
| 🌡️ **Basin temperature** | Part 2 — PyTorch architecture | Finish the half-built multimodal predictor so it's wired correctly to watch the basin. |
| ♻️ **Waste machines** | Part 3 — clustering (HW3) | Cluster tonight's waste, choose how many machines to run, configure them, and route arrivals. |

The day plays out **step by step** in the central reactor display. Each step
runs your solution against hidden safety checks. **The first check that fails
melts the plant down** — and a dashboard recaps exactly where and why. Survive
all three subsystems and you make it to dawn. 🏁

## Quick start (with [uv](https://docs.astral.sh/uv/))

The front end is pre-built and committed, so you only need Python + `uv`. From
the **repo root**:

```bash
uv run --project workshop python workshop/launch.py
```

On the first run `uv` builds an isolated environment from
`workshop/pyproject.toml` + `workshop/uv.lock`, then launches the backend and
opens your browser at <http://127.0.0.1:8000>. Press **▶ Start shift**.

> Don't have uv? Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`
> (or `brew install uv`). Prefer plain pip? `pip install -r workshop/requirements.txt`
> then `python workshop/launch.py` works too.

Out of the box every subsystem runs a bundled **reference** solution, so the
panel works before you've finished any homework.

## Plugging in your own work

1. Finish a homework notebook and run its **export** cell — it writes a
   `solution_part{N}.py` (plus any trained model artifacts).
2. Drop that file into [`workshop/solutions/`](solutions/README.md).
3. Relaunch (or just **Restart** the shift). The matching panel now badges
   **student** and runs *your* code.

See `workshop/solutions/README.md` for the exact interface each file must
expose, and `workshop/backend/reference/` for complete working examples.

## How it works

```
workshop/
  launch.py            one command → starts the backend + opens the browser
  pyproject.toml       backend deps for uv (uv.lock pins exact versions)
  requirements.txt     same deps, for pip users
  backend/
    contracts.py       the Part-1/2/3 solution protocols + StepEvent
    loader.py          picks your solutions/ file, else the reference
    engine.py          DayEngine.run() → streams the day as StepEvents
    parts/             per-subsystem adapters + the HIDDEN safety checks
    reference/         bundled working solutions (the fallback)
    datagen/           synthetic pump & basin datasets
  frontend/            Vite + React control panel (dist/ is committed)
  solutions/           ← you drop your exported solutions here (git-ignored)
```

The backend exposes:

- `GET /api/state` — which solution (student/reference) is active per part.
- `WS  /ws/run-day` — streams the whole day as `StepEvent` JSON, one beat at a
  time, so the panel animates live.

### Developing the front end

```bash
cd workshop/frontend
npm install
npm run dev          # http://localhost:5173, talks to the backend on :8000
```

In a second terminal run the backend with reload:

```bash
uv run --project workshop uvicorn workshop.backend.main:app --reload   # from the repo root
```

Rebuild the committed bundle with `npm run build` when you're done.

## Trying the failure modes

Want to see how the panel reacts when a solution is **missing**, **wrong**, or
just **bad**? There's a scenario library for exactly that:

```bash
uv run --project workshop python workshop/scenario.py list        # see all scenarios
uv run --project workshop python workshop/scenario.py use p2_wrong # drop a broken Part-2 solution in
uv run --project workshop python workshop/scenario.py reset         # clear it again
```

After `use`, launch (or hit ↻ Restart) and watch where it melts down. To check
all nine failure modes at once, headlessly:

```bash
uv run --project workshop python workshop/tests/run_scenarios.py
```

See `workshop/tests/README.md` for the full 3×3 matrix.

## The game rules

- **Pump** melts down if the model misses too many real failures (recall too
  low) or floods the operator with false alarms (precision too low).
- **Basin** melts down if the architecture is wrong — it won't run, has no
  regression head, leaves the vision encoder un-frozen, or has a branch that
  never reaches the prediction. (Graded structurally — no training.)
- **Waste** melts down if the clustering is meaningless (silhouette too low) or
  routing sends a batch to a machine that doesn't exist.

The hidden thresholds live in `backend/parts/part{1,2,3}_*.py` — your solution
never sees them, so it has to actually work. Good luck, Manager. ☢️
