"""Reference solution — Part 1: Coolant-pump failure prediction.

Mirrors the worked answer exported by ``HW1_pump_cleaning_fitting_solution.ipynb``
(the notebook's "Wrap-up" cell), so the bundled fallback behaves exactly like a
correct student submission against the same hidden checks::

    sol = get_solution()
    clean_df = sol.clean(raw_log)     # tidy a messy log (no NaNs / dupes / junk)
    preds    = sol.predict(raw_log)   # 0/1 failure-within-the-hour, one per row

The only change from the exported file is the import: inside the backend package
we pull the shared data generator with a package-relative import instead of the
notebook's ``sys.path`` shim. The schema (``Pump_ID``, ``pump_failure_status``,
full sensor names) is owned by :mod:`backend.datagen.pump_gen`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, RANSACRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..datagen import pump_gen

SENSORS = pump_gen.SENSORS
TARGET = pump_gen.TARGET
PUMP_ID_COL = pump_gen.PUMP_ID_COL
NUMERIC_FEATURES = ["inspection_rating", "runtime_hours", *SENSORS]
CATEGORICAL_FEATURES = [PUMP_ID_COL]
MISSING_SENSOR_COLS = [
    "vibration_amplitude",
    "bearing_temperature",
    "motor_current",
    "motor_voltage",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RATING_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
# Plausible physical ranges; readings outside become NaN (sensor glitches).
RANGES = {
    "flow_rate": (50.0, 900.0),
    "inlet_pressure": (0.5, 3.0),
    "outlet_pressure": (1.0, 10.0),
    "vibration_amplitude": (0.0, 80.0),
    "bearing_temperature": (290.0, 500.0),
    "motor_current": (0.0, 220.0),
    "motor_voltage": (300.0, 460.0),
}


def _format_pump_id(value):
    """Canonicalise ids like ``pmp-001``/``PMP001``/``PMP_001`` → ``PMP-001``."""
    digits = "".join(ch for ch in str(value).upper() if ch.isdigit())
    if not digits:
        return np.nan
    return f"PMP-{int(digits):03d}"


def _fit_ransac_keep_mask(train_df, target_col=TARGET, residual_scale=8.0):
    """Robust per-sensor RANSAC fit on the healthy class → keep-mask of inliers."""
    keep = np.ones(len(train_df), dtype=bool)
    for label in [0]:
        idx = train_df.index[train_df[target_col] == label]
        subset = train_df.loc[idx]
        class_keep = np.ones(len(subset), dtype=bool)
        for sensor in MISSING_SENSOR_COLS:
            predictors = [c for c in NUMERIC_FEATURES if c != sensor]
            model = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=0.65,
                random_state=0,
            )
            model.fit(subset[predictors], subset[sensor])
            residual = np.abs(subset[sensor] - model.predict(subset[predictors]))
            med = np.median(residual)
            mad = np.median(np.abs(residual - med))
            robust_sigma = 1.4826 * (
                mad if mad > 0 else np.std(residual) if np.std(residual) > 0 else 1.0
            )
            threshold = med + residual_scale * robust_sigma
            class_keep &= ~(residual > threshold).to_numpy()
        keep[idx] = class_keep
    return keep


class PumpSolution:
    def __init__(self) -> None:
        self._medians: dict[str, float] = {}
        self._model: Pipeline | None = None
        self._fit_on_history()

    # ---- the cleaning pipeline (Part 1's core learning objective) ---------- #
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if PUMP_ID_COL in out.columns:
            out[PUMP_ID_COL] = out[PUMP_ID_COL].map(_format_pump_id)
        if "inspection_rating" in out.columns:
            out["inspection_rating"] = out["inspection_rating"].map(
                lambda v: RATING_WORDS[v] if v in RATING_WORDS else v
            )
            out["inspection_rating"] = pd.to_numeric(out["inspection_rating"], errors="coerce")

        for col in NUMERIC_FEATURES:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # impossible sensor readings → NaN so they get imputed
        for sensor, (lo, hi) in RANGES.items():
            if sensor not in out.columns:
                continue
            out.loc[(out[sensor] < lo) | (out[sensor] > hi), sensor] = np.nan

        # impute with TRAINING medians (fall back to column median if unknown)
        for col in NUMERIC_FEATURES:
            if col not in out.columns:
                continue
            med = self._medians.get(col, out[col].median())
            out[col] = out[col].fillna(med)

        return out.drop_duplicates().reset_index(drop=True)

    # ---- features + prediction -------------------------------------------- #
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted.")
        clean = self.clean(df)
        return self._model.predict(clean[ALL_FEATURES]).astype(int)

    def _make_model(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        [
                            ("num", StandardScaler(), NUMERIC_FEATURES),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                        ]
                    ),
                ),
                (
                    "model",
                    LogisticRegression(max_iter=200, random_state=0, class_weight="balanced"),
                ),
            ]
        )

    # ---- training (on the historical log) --------------------------------- #
    def _fit_on_history(self) -> None:
        history = pump_gen.generate(seed=0)
        messy = pump_gen.make_messy(history.drop(columns=["failure_mode"]), seed=11)
        cleaned = self.clean(messy)
        # lock in the training medians, then re-clean with them in place
        self._medians = {col: float(cleaned[col].median()) for col in NUMERIC_FEATURES}
        cleaned = self.clean(messy)
        keep = _fit_ransac_keep_mask(cleaned)
        train_df = cleaned.loc[keep].reset_index(drop=True)
        self._model = self._make_model()
        self._model.fit(train_df[ALL_FEATURES], train_df[TARGET].astype(int))


def get_solution() -> PumpSolution:
    return PumpSolution()
