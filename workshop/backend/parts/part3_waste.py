"""Part 3 adapter — Waste-machine clustering (HW3).

Narrative: cluster tonight's waste to find how many machines we need (silhouette
+ elbow), configure each machine's settings from its cluster, then route the
incoming batches into those machines. The plant melts down if the clustering is
junk (too few machines / a meaningless silhouette) or routing sends waste to a
machine that doesn't exist.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, silhouette_score

from . import exploded, ok, running

# Reuse the real HW3 data helper (new home first, legacy fallback).
for _rel in ("../../homework/helpers", "../../../exercises"):
    _cand = os.path.abspath(os.path.join(os.path.dirname(__file__), _rel))
    if os.path.exists(os.path.join(_cand, "hw3_data.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
import hw3_data  # noqa: E402

PART = 3
SILHOUETTE_MIN = 0.35       # below this the "clusters" are basically noise
K_RANGE = range(2, 9)
ROUTE_AGREEMENT_MIN = 0.90  # routing must match "nearest configured machine"
IMPUTE_R2_MIN = 0.50        # filled quantities must be this trustworthy (regression)
IMPUTE_ACC_MIN = 0.65       # filled critical-flags must beat a lazy guess (base rate ≈ 0.49)
NEXT_DAY_N = 250            # tomorrow's intake — enough scored cells for stable R²/acc,
                            # and the sample size where the silhouette cleanly recovers k = 5


def _level(value, series):
    if value >= series.quantile(0.66):
        return "HIGH"
    if value <= series.quantile(0.33):
        return "LOW"
    return "MED"


def run(solution):
    # ---- 0. repair tomorrow's incomplete intake log ------------------------
    history = hw3_data.load_history(verbose=False)                                   # complete training log
    gappy = hw3_data.load_next_day(n=NEXT_DAY_N, verbose=False)                       # tomorrow, with gaps
    truth = hw3_data.load_next_day(n=NEXT_DAY_N, missing_frac=0.0, verbose=False)     # oracle (same seed, unmasked)
    qcol, ccol = hw3_data.QUANTITY_COL, hw3_data.CRITICAL_COL
    n_missing_q = int(gappy[qcol].isna().sum())
    n_missing_c = int(gappy[ccol].isna().sum())

    yield running(PART, "waste.repair", "Repairing tomorrow's log",
                  f"Tomorrow's intake arrived with gaps — {n_missing_q} missing quantities and "
                  f"{n_missing_c} missing critical-flags. Training RandomForests on "
                  f"{len(history)} batches of history to fill them before anything else.",
                  {"missing_quantity": n_missing_q, "missing_critical": n_missing_c,
                   "rows": len(gappy)})
    try:
        filled = solution.fill_gaps(history, gappy)
    except Exception as exc:  # noqa: BLE001
        yield exploded(PART, "waste.repair", "Log repair crashed",
                       f"fill_gaps() raised {type(exc).__name__}: {exc}")
        return
    if filled[[qcol, ccol]].isna().any().any():
        yield exploded(PART, "waste.repair", "Gaps left in the log",
                       "Missing quantities or critical-flags remain after fill_gaps() — "
                       "we can't plan on an incomplete log.")
        return

    qmask = gappy[qcol].isna().to_numpy()
    cmask = gappy[ccol].isna().to_numpy()
    r2 = float(r2_score(truth.loc[qmask, qcol], filled.loc[qmask, qcol])) if qmask.any() else 1.0
    acc = (float(accuracy_score(truth.loc[cmask, ccol].astype(int),
                                filled.loc[cmask, ccol].astype(int))) if cmask.any() else 1.0)
    repair_payload = {"missing_quantity": n_missing_q, "missing_critical": n_missing_c,
                      "r2": round(r2, 3), "accuracy": round(acc, 3),
                      "r2_min": IMPUTE_R2_MIN, "acc_min": IMPUTE_ACC_MIN}
    if r2 < IMPUTE_R2_MIN:
        yield exploded(PART, "waste.repair", "Quantities filled with garbage",
                       f"Imputed quantity R² {r2:.2f} < {IMPUTE_R2_MIN:.2f}: the filled amounts "
                       "are unreliable, so the whole plan would rest on bad data.", repair_payload)
        return
    if acc < IMPUTE_ACC_MIN:
        yield exploded(PART, "waste.repair", "Critical flags guessed wrong",
                       f"Imputed critical-flag accuracy {acc:.0%} < {IMPUTE_ACC_MIN:.0%}: hazardous "
                       "waste would be mislabelled and mishandled.", repair_payload)
        return
    yield ok(PART, "waste.repair", "Tomorrow's log repaired",
             f"Filled {n_missing_q} quantities (R² {r2:.2f}) and {n_missing_c} critical-flags "
             f"({acc:.0%} accurate). Planning on the repaired log.", repair_payload)

    waste = filled   # everything below now plans on the repaired intake

    # ---- 1. find the number of machines ------------------------------------
    yield running(PART, "waste.search", "Sorting tonight's waste",
                  f"Clustering {len(waste)} waste batches and trying 2–8 machines. The "
                  "silhouette score rewards tight, well-separated groups; the inertia elbow "
                  "shows where adding machines stops helping. We pick the sweet spot.",
                  {"batches": len(waste)})
    try:
        plan = solution.build_machine_plan(waste)
    except Exception as exc:  # noqa: BLE001
        yield exploded(PART, "waste.search", "Clustering crashed",
                       f"build_machine_plan() raised {type(exc).__name__}: {exc}")
        return
    if not all(key in plan for key in ("k", "km", "X", "data", "prep")):
        yield exploded(PART, "waste.search", "Incomplete machine plan",
                       "build_machine_plan() must return at least k, km, X, data and prep.")
        return

    X = np.asarray(plan["X"])
    best_k = int(plan["k"])
    if best_k < 3:
        yield exploded(PART, "waste.search", "Only one machine for everything",
                       f"k = {best_k}: mixing every waste type into one machine "
                       "guarantees a bad reaction.", {"best_k": best_k})
        return

    ks, sils, inertias = [], [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        ks.append(k)
        sils.append(round(float(silhouette_score(X, km.labels_)), 3))
        inertias.append(round(float(km.inertia_), 1))
    best_sil = float(silhouette_score(X, plan["km"].labels_))

    search_payload = {"ks": ks, "silhouettes": sils, "inertias": inertias,
                      "best_k": best_k, "best_sil": round(best_sil, 3),
                      "silhouette_min": SILHOUETTE_MIN}
    if best_sil < SILHOUETTE_MIN:
        yield exploded(PART, "waste.search", "Clusters are meaningless → mis-sorted waste",
                       f"Silhouette {best_sil:.2f} < {SILHOUETTE_MIN:.2f}: the groups "
                       "overlap, so machines get the wrong waste.", search_payload)
        return
    yield ok(PART, "waste.search", f"{best_k} machines is the sweet spot",
             f"Silhouette peaks at k = {best_k} ({best_sil:.2f}).", search_payload)

    # ---- 2. configure each machine -----------------------------------------
    yield running(PART, "waste.configure", "Configuring the machines",
                  "Reading each cluster's average profile to set that machine's dials — "
                  "shielding, heat and half-life — so it's tuned to the waste it will handle.")
    data = plan["data"]
    prof = data.groupby("machine")[hw3_data.PHYSICAL_FEATURES].mean()
    load = data.groupby("machine")[hw3_data.QUANTITY_COL].sum()
    chem = (data.groupby("machine")[hw3_data.CATEGORICAL[0]]
            .agg(lambda s: s.value_counts().index[0]))
    # if the (repaired) log carries critical flags, a machine is "dangerous" when
    # it mostly handles critical waste — otherwise fall back to high shielding.
    has_crit = hw3_data.CRITICAL_COL in data.columns
    crit = data.groupby("machine")[hw3_data.CRITICAL_COL].mean() if has_crit else None
    machines = []
    for m in prof.index:
        shielding = _level(prof["radioactivity_Bq"][m], prof["radioactivity_Bq"])
        card = {
            "id": int(m),
            "load_kg": int(load[m]),
            "shielding": shielding,
            "temperature": _level(prof["heat_output_W"][m], prof["heat_output_W"]),
            "half_life": _level(prof["half_life_years"][m], prof["half_life_years"]),
            "chemical": str(chem[m]),
            "count": int((data["machine"] == m).sum()),
            "danger": (float(crit[m]) >= 0.5) if has_crit else (shielding == "HIGH"),
        }
        if has_crit:
            card["critical_share"] = round(float(crit[m]) * 100)
        machines.append(card)
    yield ok(PART, "waste.configure", f"{best_k} machines configured",
             f"Total {int(load.sum())} kg queued across {best_k} machines.",
             {"machines": machines, "k": best_k, "total_kg": int(load.sum())})

    # ---- 3. route the incoming waste ---------------------------------------
    yield running(PART, "waste.route", "Routing incoming waste",
                  "Matching each new batch to its nearest configured machine — reusing the "
                  "clustering we just fitted, not re-computing it. (★ = tonight's arrivals)")
    incoming = hw3_data.load_incoming(verbose=False)
    try:
        ids = np.asarray(solution.route(plan, incoming)).astype(int).ravel()
    except Exception as exc:  # noqa: BLE001
        yield exploded(PART, "waste.route", "Routing crashed",
                       f"route() raised {type(exc).__name__}: {exc}")
        return
    if len(ids) != len(incoming) or ids.min() < 0 or ids.max() >= best_k:
        yield exploded(PART, "waste.route", "Waste sent to a non-existent machine",
                       f"route() returned ids outside 0..{best_k - 1} (or the wrong "
                       "count) — that batch has nowhere to go.")
        return

    # Correctness oracle: the ONE correct routing is "each batch → its nearest
    # configured machine", i.e. reuse the fitted prep + km. We recompute that
    # ourselves and compare, so a solution that re-clusters the arrivals or
    # assigns them ad-hoc is caught even though its ids look valid.
    inc_feats = incoming[hw3_data.PHYSICAL_FEATURES + hw3_data.CATEGORICAL]
    try:
        expected = np.asarray(plan["km"].predict(
            np.asarray(plan["prep"].transform(inc_feats)))).astype(int)
    except Exception as exc:  # noqa: BLE001
        yield exploded(PART, "waste.route", "Routing can't be verified",
                       f"The plan's fitted preprocessor/clusterer couldn't score the new "
                       f"batches ({type(exc).__name__}). Return the fitted prep + km in the "
                       "plan and route the arrivals with them.")
        return
    agreement = float(np.mean(ids == expected))
    if agreement < ROUTE_AGREEMENT_MIN:
        yield exploded(PART, "waste.route", "Waste mis-routed → wrong machine",
                       f"Only {agreement:.0%} of arrivals reached their nearest configured "
                       "machine. Reuse the fitted pipeline (predict) — don't re-cluster the "
                       "new batches or assign them by hand.",
                       {"agreement": round(agreement, 3),
                        "agreement_min": ROUTE_AGREEMENT_MIN})
        return

    # 2-D PCA scatter (best-effort viz): existing batches + incoming (stars)
    train_points, incoming_points = [], []
    pca = plan.get("pca")
    if pca is not None:
        try:
            train_xy = pca.transform(X)
            inc_xy = pca.transform(np.asarray(plan["prep"].transform(inc_feats)))
            sel = np.linspace(0, len(train_xy) - 1, min(220, len(train_xy))).round().astype(int)
            train_points = [{"x": round(float(train_xy[i, 0]), 2), "y": round(float(train_xy[i, 1]), 2),
                             "m": int(data["machine"].iloc[i])} for i in sel]
            incoming_points = [{"x": round(float(inc_xy[i, 0]), 2), "y": round(float(inc_xy[i, 1]), 2),
                                "m": int(ids[i])} for i in range(len(ids))]
        except Exception:  # noqa: BLE001 — the scatter is optional eye-candy
            train_points, incoming_points = [], []
    counts = {int(m): int((ids == m).sum()) for m in range(best_k)}

    yield ok(PART, "waste.route", f"Routed {len(ids)} new batches",
             f"All {len(ids)} arrivals reached their nearest machine ({agreement:.0%} match).",
             {"counts": counts, "train_points": train_points,
              "incoming_points": incoming_points, "k": best_k,
              "agreement": round(agreement, 3)})
