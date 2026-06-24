"""Part 1 adapter — Coolant-pump failure prediction.

Narrative: read the messy logs → scrub them → score failure risk for the next
hour. The plant melts down if the model misses too many real failures (low
recall) or cries wolf constantly (low precision).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from ..datagen import pump_gen
from . import downsample, exploded, ok, running

PART = 1
RECALL_MIN = 0.60     # must catch most real failures
PRECISION_MIN = 0.30  # …without drowning the operator in false alarms


def _preview(df, n=6):
    cols = [pump_gen.PUMP_ID_COL, "flow_rate", "vibration_amplitude",
            "bearing_temperature", "motor_current"]
    cols = [c for c in cols if c in df.columns]
    rec = df[cols].head(n).to_dict(orient="records")
    out = []
    for r in rec:
        out.append({k: (None if (isinstance(v, float) and np.isnan(v)) else
                        (round(v, 1) if isinstance(v, (int, float)) else v))
                    for k, v in r.items()})
    return out


def run(solution):
    # ---- 1. read the raw historical log -------------------------------------
    hist = pump_gen.make_messy(pump_gen.generate(seed=0).drop(columns=["failure_mode"]), seed=3)
    nan_before = int(hist[pump_gen.SENSORS].isna().sum().sum())
    dup_before = int(hist.duplicated().sum())
    yield running(PART, "pump.load", "Reading the coolant-pump logs",
                  f"Pulling {len(hist):,} telemetry rows from last shift. Sensors dropped "
                  f"offline ({nan_before:,} blank cells), rows got duplicated ({dup_before}), "
                  "and formats drift — a real, messy log.",
                  {"rows": len(hist), "nan_cells": nan_before, "dup_rows": dup_before,
                   "failures": int(hist[pump_gen.TARGET].sum()), "preview": _preview(hist)})

    # ---- 2. clean it --------------------------------------------------------
    yield running(PART, "pump.clean", "Scrubbing the logs",
                  "Imputing the gaps, dropping duplicate rows, and turning impossible "
                  "sensor spikes into gaps too — a model can't learn from NaNs or junk.")
    try:
        cleaned = solution.clean(hist)
    except Exception as exc:  # noqa: BLE001
        yield exploded(PART, "pump.clean", "Cleaning crashed",
                       f"clean() raised {type(exc).__name__}: {exc}")
        return
    nan_after = int(cleaned[pump_gen.SENSORS].isna().sum().sum()) if len(cleaned) else 0
    if nan_after > 0:
        yield exploded(PART, "pump.clean", "Dirty data reached the model",
                       f"{nan_after} missing sensor values survived cleaning — the "
                       "model can't run on NaNs.",
                       {"nan_before": nan_before, "nan_after": nan_after})
        return
    yield ok(PART, "pump.clean", "Logs cleaned",
             f"{len(hist)} → {len(cleaned)} rows · {nan_before} gaps filled · "
             f"{len(hist) - len(cleaned)} duplicates dropped.",
             {"rows_before": len(hist), "rows_after": len(cleaned),
              "nan_before": nan_before, "nan_after": nan_after,
              "dupes_removed": len(hist) - len(cleaned)})

    # ---- 3. score the next hour on a held-out deployment set ----------------
    yield running(PART, "pump.predict", "Scoring failure risk",
                  "Feeding fresh readings to your model to flag a failure one hour ahead. "
                  "What matters: recall (catch the real failures) without flooding the "
                  "operator with false alarms (precision).")
    ev = pump_gen.generate(seed=1)
    y_true = ev[pump_gen.TARGET].to_numpy()
    raw = pump_gen.make_messy(ev.drop(columns=[pump_gen.TARGET, "failure_mode"]),
                              seed=5, add_duplicates=False)
    try:
        out = np.asarray(solution.predict(raw)).ravel()
    except Exception as exc:  # noqa: BLE001
        yield exploded(PART, "pump.predict", "Prediction crashed",
                       f"predict() raised {type(exc).__name__}: {exc}")
        return
    if len(out) != len(y_true):
        yield exploded(PART, "pump.predict", "Bad prediction output",
                       f"predict() must return one value per row "
                       f"(got {len(out)} values for {len(y_true)} rows).")
        return
    if not np.all(np.isin(out, (0, 1))):
        yield exploded(PART, "pump.predict", "Bad prediction output",
                       "predict() must return hard 0/1 labels, not probabilities or "
                       "scores — did you return predict_proba() output by mistake?")
        return
    preds = out.astype(int)

    recall = float(recall_score(y_true, preds, zero_division=0))
    precision = float(precision_score(y_true, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    # an hourly risk curve for the panel: bucket the shift into 24 "hours"
    chunks = np.array_split(np.arange(len(preds)), 24)
    risk = [float(preds[c].mean()) for c in chunks]
    real = [float(y_true[c].mean()) for c in chunks]

    payload = {"recall": round(recall, 3), "precision": round(precision, 3),
               "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
               "flagged": int(preds.sum()), "actual": int(y_true.sum()),
               "risk_curve": risk, "real_curve": real,
               "recall_min": RECALL_MIN, "precision_min": PRECISION_MIN}

    if recall < RECALL_MIN:
        yield exploded(PART, "pump.predict", "Pump failure went undetected → CORE OVERHEAT",
                       f"Recall {recall:.0%} < {RECALL_MIN:.0%}: {fn} real failures "
                       "slipped through and the coolant pump seized.", payload)
        return
    if precision < PRECISION_MIN:
        yield exploded(PART, "pump.predict", "Too many false alarms → manual override chaos",
                       f"Precision {precision:.0%} < {PRECISION_MIN:.0%}: {fp} false alarms "
                       "masked the real one. Boom.", payload)
        return
    yield ok(PART, "pump.predict", "Pump under control",
             f"Caught {tp}/{tp + fn} failures (recall {recall:.0%}, "
             f"precision {precision:.0%}).", payload)
