"""Reference solution — Part 3: Waste-machine clustering (HW3).

Lifts the worked answer straight out of
``exercises/HW3_clustering_waste_solution.ipynb``: ``build_machine_plan`` runs
feature selection → log/scale/one-hot preprocessing → a silhouette-driven
search for the number of machines → a fitted KMeans; ``route`` sends new
arrivals into the already-fitted machines without re-fitting.

The real HW3 data helper (``exercises/hw3_data.py``) is reused as-is.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Make the course's HW3 data helper importable (new home first, legacy fallback).
_HERE = os.path.dirname(__file__)
for _rel in ("../../homework/helpers", "../../../exercises"):
    _cand = os.path.abspath(os.path.join(_HERE, _rel))
    if os.path.exists(os.path.join(_cand, "hw3_data.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break

import hw3_data  # noqa: E402  (path injected above)


class WasteSolution:
    def fill_gaps(self, history, gappy):
        """HW3 Part 7: train on the complete history, fill tomorrow's gaps.

        Missing ``quantity_kg`` → RandomForest regression; missing ``critical``
        flags → RandomForest classification; both predicted from the physical
        features, filling only the blank cells.
        """
        feat = hw3_data.PHYSICAL_FEATURES
        out = gappy.copy()

        qcol = hw3_data.QUANTITY_COL
        reg = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
        reg.fit(history[feat], history[qcol])
        qgap = out[qcol].isna()
        if qgap.any():
            out.loc[qgap, qcol] = reg.predict(out.loc[qgap, feat])

        ccol = hw3_data.CRITICAL_COL
        known = history[ccol].notna()
        clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        clf.fit(history.loc[known, feat], history.loc[known, ccol].astype(int))
        cgap = out[ccol].isna()
        if cgap.any():
            out.loc[cgap, ccol] = clf.predict(out.loc[cgap, feat])
        out[ccol] = out[ccol].astype(int)
        return out

    def build_machine_plan(self, raw_df, k_range=range(2, 9)) -> Dict[str, Any]:
        """Parts 2-5 of HW3, packaged: machines fitted on ``raw_df``."""
        feats = raw_df[hw3_data.PHYSICAL_FEATURES + hw3_data.CATEGORICAL]

        prep = ColumnTransformer([
            ("heavy", Pipeline([("log", FunctionTransformer(np.log10)),
                                ("scale", StandardScaler())]),
             hw3_data.LOG_FEATURES),
            ("tame", StandardScaler(),
             [c for c in hw3_data.PHYSICAL_FEATURES if c not in hw3_data.LOG_FEATURES]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), hw3_data.CATEGORICAL),
        ])
        Xp = np.asarray(prep.fit_transform(feats))

        ks = list(k_range)
        sils = [silhouette_score(Xp, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xp))
                for k in ks]
        k = ks[int(np.argmax(sils))]
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xp)

        out = raw_df.copy()
        out["machine"] = km.labels_
        return {"k": k, "prep": prep, "km": km, "pca": PCA(2, random_state=0).fit(Xp),
                "X": Xp, "data": out}

    def route(self, plan: Dict[str, Any], new_df) -> np.ndarray:
        """Send each new batch to an existing machine (no re-fitting)."""
        feats = new_df[hw3_data.PHYSICAL_FEATURES + hw3_data.CATEGORICAL]
        routing = Pipeline([("prep", plan["prep"]), ("kmeans", plan["km"])])
        return np.asarray(routing.predict(feats))


def get_solution() -> WasteSolution:
    return WasteSolution()
