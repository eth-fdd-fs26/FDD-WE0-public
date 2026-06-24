# ☢️ Nuclear Central Manager — how the simulation works

A precise, end-to-end walkthrough of the core project: the data flow, the
solution contract, the day engine, each of the three subsystems, and the front
end. Everything below is grounded in the actual files under `workshop/`.

---

## 1. The big picture: one data flow

```
React (frontend/)  ──WebSocket /ws/run-day?pace=X──►  FastAPI (backend/main.py)
      ▲                                                      │
      │  StepEvent JSON, one beat at a time                  ▼
      └──────────────────────────────────────────  DayEngine.run()  (engine.py)
                                                            │  for each part:
                                                            ▼
                                          loader.load_solution(part)  (loader.py)
                                                            │  picks solutions/ or reference/
                                                            ▼
                                          parts/partN_*.run(solution)  ← the adapter
                                                            │  runs solution + HIDDEN checks
                                                            ▼
                                          yields StepEvent(... status, payload ...)
```

The browser never computes anything about the ML — it just **renders events**.
All logic runs in Python and is streamed as small JSON `StepEvent`s.

A `StepEvent` (defined in `backend/contracts.py`) is:

```python
StepEvent(part, step_id, title, status, message, payload)
# status ∈ {running, ok, exploded, info}
```

- `running` = "I'm working on this" (carries no result data yet)
- `ok` = "passed the safety check" (carries the result payload the panel draws)
- `exploded` = "failed → meltdown"
- `info` = narrative beat (shift start, briefings, dawn)

---

## 2. The solution contract + student vs reference

`contracts.py` declares three `Protocol`s — the *only* thing a solution must
implement:

| Part | Methods the solution must expose |
|------|----------------------------------|
| 1    | `clean(df) -> df`, `predict(df) -> 0/1 array` |
| 2    | `build_model() -> nn.Module` (a fresh, untrained model — graded structurally) |
| 3    | `fill_gaps(history, gappy) -> df`, `build_machine_plan(df) -> dict`, `route(plan, new_df) -> ids` |

Every solution file also exposes `get_solution()` returning that object.

`loader.load_solution(part)`:

1. If `workshop/solutions/solution_part{N}.py` exists → import it by file path
   (`importlib.util.spec_from_file_location`), tag source `"student"`.
2. Otherwise import `backend.reference.solution_part{N}`, tag source `"reference"`.
3. Call `module.get_solution()`, return `(object, source)`.

That `source` string is what the panel badges show ("student"/"reference").
**Crucially, the safety checks are NOT in the solution** — they live in the
adapters, so a solution can't see or fake them.

---

## 3. The engine + pacing (`engine.py`)

`DayEngine.run()` is an async generator. It:

1. Yields `info day.start`.
2. For each `(part, label, module)` in `[pump, basin, waste]`:
   - `load_solution(part)` (a failed import → instant explode for that part).
   - Yields a one-line **briefing** (`PART_INTRO[part]`, the "why").
   - Iterates `module.run(solution)` and re-yields each `StepEvent`. The moment
     one has `status == "exploded"`, it yields a `day.meltdown` beat and
     **returns** (instant-explode model).
3. If all three finish → `info day.victory`.

**Pacing** is `_dwell(ev)`: after each event it sleeps `pace × 1.7` for
`ok`/`info` (results linger so you can read), `× 0.85` for `running` (quick),
`× 1.2` for `exploded`. `pace` comes from the WebSocket query param the front
end sends (Slow=1.1, Normal=0.6, Fast=0.3).

---

## 4. Part 1 — Coolant pump (`parts/part1_pump.py` + `datagen/pump_gen.py`)

**The data.** `pump_gen.generate(seed)` builds a 10,000-row telemetry log,
exactly 5% failures. Four pumps PMP-001..004 with intrinsic inspection rating +
base runtime + failure weight (older = fails more). Healthy rows get
baseline-normal sensor readings; each failure row gets one of three mode
signatures:

- **mechanical** (50%): vibration ×~3, bearing_temp +~28 °C
- **hydraulic** (30%): flow_rate ×0.45, motor_current ×0.6 (cavitation)
- **stall** (20%): outlet_pressure ×0.4, motor_current ×1.8 (spike)

`make_messy()` then dirties it: ~4% cells set to NaN, sensor glitches
(`flow_rate=0`, `bearing_temp=999`), inconsistent id casing (`" pmp-001 "`), and
~2% duplicate rows.

**The three streamed steps:**

1. `pump.load` — reads `make_messy(generate(0))`, reports rows / NaN cells /
   dup rows / a preview.
2. `pump.clean` — calls `solution.clean(messy_hist)`. **Check:** zero NaNs
   remain in the sensor columns. Fail → explode *"Dirty data reached the
   model"*. The `ok` payload reports rows before/after, gaps filled, dupes
   removed.
3. `pump.predict` — the scored step. Builds a **held-out deployment set**
   `generate(seed=1)` (different seed = no leakage), keeps the true labels, then
   `make_messy(..., add_duplicates=False)` so the row count stays aligned. Calls
   `solution.predict(raw)`, validates output is 0/1 of the right length, computes
   **recall, precision, confusion matrix**, and a 24-bucket "hourly risk" curve.

**Explosion conditions:** `recall < 0.60` → *"Pump failure went undetected →
CORE OVERHEAT"*; or `precision < 0.30` → *"Too many false alarms"*. Otherwise
`ok`.

**The reference solution (`reference/solution_part1.py`).** `clean()`
canonicalises `pump_id`, turns out-of-physical-range readings into NaN, imputes
with **training medians**, drops duplicates. It fits a
`RandomForestClassifier(200, class_weight="balanced")` on the cleaned historical
log; features = the 7 sensors + inspection_rating + runtime_hours + one-hot pump
id. It scores recall ≈ 0.99, precision ≈ 1.0 — comfortably over the thresholds.

---

## 5. Part 2 — Basin temperature (`parts/part2_basin.py` + `homework/helpers/basin_lab.py`)

This part wires straight into the real HW2 notebook (like Part 3 does with
HW3). The adapter adds `homework/helpers` to `sys.path` and imports `basin_lab`
(the *given* building blocks: `SpectroNet`, `VisionEncoder`, the feature-width
constants). HW2 is an **architecture** exercise — the student finishes a
half-built multimodal predictor — so Part 2 is graded **structurally, with no
training and no saved weights.** The solution only hands back the architecture
via `build_model() -> nn.Module`.

**The model.** `BasinNet(x_sig, x_th)` fuses a **1-D signal branch**
(`SpectroNet` with its classifier sliced off → `(B, 32)` features) with a
**frozen thermal vision encoder** (frames reshaped via einops → `(B, 32)`),
concatenates to `(B, 64)`, passes through a **residual MLP core**, and ends in a
**regression head** → one temperature per sample. Inputs are
`x_sig (B, 4, 64)` and `x_th (B, 3, 16, 16, 1)`.

**The two streamed steps (both instant — just forward/backward on dummy data):**

1. `basin.arch` — builds `solution.build_model()` and probes shapes:
   the **signal branch** must output `(B, 32)` features, not class logits
   (explode *"…still ends in the classifier"*); the **residual core** must
   preserve its width so the skip lines up (*"Residual skip can't line up"*);
   the **whole model** must run and end in one temperature per sample — a crash
   → *"Network won't run"* (flipped size / wrong fusion width), a `(B, 64)`
   output → *"No regression head"*. The `ok` payload carries the check list +
   trainable/frozen param counts and the `32+32→64→1` fusion shape.
2. `basin.wiring` — on the same model: the vision **encoder must be frozen**
   (else *"Vision encoder isn't frozen"*); perturbing the thermal input must
   **change the prediction** (else *"Thermal branch is ignored"* — the
   shape-only checks can't catch a branch that's dropped from `forward`); and a
   single `loss.backward()` must put **non-zero gradients on every trainable
   block** (signal extractor, core, head) — a dead block → *"…isn't wired into
   the output"*.

There is **no forecast/MAE step** — Part 2's safety check is *"is the backup
predictor wired correctly?"*, which matches the notebook's own framing (the
trained predictor is "handed back"; the plant only verifies the architecture).

**The reference (`reference/solution_part2.py`)** is the notebook's finished
architecture lifted verbatim: signal extractor (`Sequential(*children[:-1])`),
frozen `VisionEncoder`, `MLPCore` residual block, `Linear(64, 1)` head, exposed
through `build_model()`. It's the correct version of exactly the bugs HW2 plants
(leftover classifier, broken reshape, dropped skip, flipped size, un-frozen
encoder, missing head, ignored branch) — each one trips one of the checks above.

---

## 6. Part 3 — Waste machines (`parts/part3_waste.py`, reusing `exercises/hw3_data.py`)

This part wires straight into the real HW3 notebook. The adapter adds
`exercises/` to `sys.path` and imports `hw3_data`.

**Four streamed steps** (the first is HW3's Part 7 — repair the log before planning):

0. `waste.repair` — `history = hw3_data.load_history()` (complete) and
   `gappy = hw3_data.load_next_day()` (tomorrow's intake with ~25% of
   `quantity_kg` and `critical` blanked). Calls `solution.fill_gaps(history,
   gappy)`. **Oracle:** the *unmasked* truth is regenerated with
   `load_next_day(missing_frac=0.0)` (same seed), so on the masked cells the
   adapter scores **R² of the filled quantities** (≥ 0.50) and **accuracy of the
   filled critical-flags** (≥ 0.65, vs a ≈0.49 base rate). Also requires zero NaNs
   left. Fail → explode *"filled with garbage"* / *"critical flags guessed wrong"*.
   The repaired log then feeds the rest of the part.
1. `waste.search` — `waste = filled` (the repaired log), then
   `plan = solution.build_machine_plan(waste)`. Validates the plan dict has
   `k, km, X, data`. Independently recomputes silhouette + inertia for
   **k = 2..8** from `plan["X"]` (so the curve shown is the adapter's, not
   something the solution self-reported). `best_k = plan["k"]`, `best_sil` =
   silhouette of the plan's labels.
   **Explosion:** `best_k < 2` → *"Only one machine for everything"*; or
   `best_sil < 0.35` → *"Clusters are meaningless → mis-sorted waste"*.
2. `waste.configure` — groups `plan["data"]` by machine; computes mean physical
   profile, total kg, dominant chemical class; turns each into HIGH/MED/LOW dials
   (by quantiles) for shielding (radioactivity), heat, half-life. Because the
   repaired log carries the `critical` flag, each machine also gets a
   **critical share**, and `danger = critical share ≥ 50%` (falling back to high
   shielding if no flags). Streams the machine cards.
3. `waste.route` — `incoming = hw3_data.load_incoming()`;
   `ids = solution.route(plan, incoming)`. **Two checks:**
   (a) *validity* — length matches and every id ∈ `0..k-1`, else explode *"Waste
   sent to a non-existent machine"*;
   (b) *correctness oracle* — the adapter independently computes the one correct
   routing, `plan["km"].predict(plan["prep"].transform(incoming))` ("each batch →
   its nearest configured machine"), and requires the solution to agree with it on
   ≥ `ROUTE_AGREEMENT_MIN` (0.90) of batches, else explode *"Waste mis-routed →
   wrong machine"*. This catches a solution that re-clusters the arrivals or
   assigns them ad-hoc even though its ids look valid.
   Then builds the PCA scatter (existing batches as dots, arrivals as ★).

**The reference (`reference/solution_part3.py`)** is the notebook's
`build_machine_plan` lifted verbatim: `ColumnTransformer` (log10+scale the
heavy-tailed features, scale the rest, one-hot the chemical class) →
silhouette-driven KMeans search → fit at best k. `route` =
`Pipeline([prep, km]).predict` — reusing the fitted clustering, not refitting.
It recovers **k = 5**, silhouette ≈ 0.58.

---

## 7. The front end (`frontend/src/`)

- **`App.jsx`** holds a `useReducer`. Each incoming event updates
  `parts[p].steps[step_id] = ev`, the part's `line` (latest message), `exploded`
  flag, and `activePart`; `day.victory`/`day.meltdown` set `status`.
- **`panelPhase(part)`** derives each panel's visual state
  (`idle → busy → alive`, or `dead`) from `activePart`/`exploded`/`status`.
- **Each panel** (`PumpPanel`, `BasinPanel`, `WastePanel`) reads a step's data
  only once it's `ok` via
  `done(id) = steps[id]?.status === "ok" ? payload : null` — that guard avoids
  reading the empty `running` payload (which otherwise crashed the render).
  Charts are hand-rolled SVG in `charts.jsx`.
- **`ReactorCore`** recolors/animates the central core from `status`.
- **`MissionLog`** renders the whole event list with per-status icons and
  per-part colors, auto-scrolling — that's the narration.
- **`MeltdownScreen`** is the end-of-day overlay; `open`/`onClose` make it
  dismissable (✕ / click-outside / "Close & inspect"), and the reducer's
  `failure` field (captured from the `exploded` event) drives the "why". A
  📋 Report button in the top bar reopens it.
- **`api/ws.js`** opens the WebSocket (same origin in prod, `:8000` from the
  `:5173` dev server) and fetches `/api/state`.

---

## 8. The student loop (why it's a "tester")

A student finishes a notebook → its export cell writes `solution_partN.py` →
they drop it in `workshop/solutions/` → on the next run `loader` picks it over
the reference, the panel badges **student**, and **their** code is run against
the same hidden checks. If their model is good, the plant survives; if not, it
melts down at the exact step that failed, with the metric that tripped it shown
in the report. The reference solutions exist so the whole day runs before
anyone has finished a thing.

---

## Quick reference — explosion thresholds

| Part | Step | Passes when | Otherwise |
|------|------|-------------|-----------|
| 1 | `pump.clean` | no NaNs left in sensor columns | "Dirty data reached the model" |
| 1 | `pump.predict` | recall ≥ 0.60 **and** precision ≥ 0.30 | "core overheat" / "false alarms" |
| 2 | `basin.arch` | signal branch → 32 feats · core preserves width · ends in one temp/sample | "…ends in the classifier" / "Residual skip can't line up" / "Network won't run" / "No regression head" |
| 2 | `basin.wiring` | encoder frozen · both branches affect output · gradients reach every block | "Vision encoder isn't frozen" / "Thermal branch is ignored" / "…isn't wired into the output" |
| 3 | `waste.repair` | no NaNs left **and** quantity R² ≥ 0.50 **and** critical accuracy ≥ 0.65 | "filled with garbage" / "critical flags guessed wrong" |
| 3 | `waste.search` | k ≥ 3 **and** silhouette ≥ 0.35 | "one machine" / "clusters meaningless" |
| 3 | `waste.route` | ids correct length & all in `0..k-1`, **and** ≥ 90% match the "nearest configured machine" oracle | "non-existent machine" / "mis-routed → wrong machine" |
