"""Synthetic coolant-pump telemetry — Part 1 ("Pump Failure Prediction").

Generation follows ``exercises/temp/build_nuclear_pump_dataset.py``; messiness
follows ``exercises/temp/corrupt_nuclear_pump_dataset.py`` so the workshop
simulator and ``data/hw_pump.csv`` share the same physics and corruption rules.

* 10,000 rows, 5% failure rate.
* Four pumps PMP-001..PMP-004 with inspection rating, base runtime, failure weight.
* Three failure modes: mechanical (50%), hydraulic (30%), stall (20%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_ROWS = 10_000
FAILURE_RATE = 0.05

TARGET = "pump_failure_status"
PUMP_ID_COL = "Pump_ID"

# Sensor columns used by the homework notebook and the workshop scorer.
SENSORS = [
    "flow_rate",
    "inlet_pressure",
    "outlet_pressure",
    "vibration_amplitude",
    "bearing_temperature",
    "motor_current",
    "motor_voltage",
]

# model, inspection_rating, base_runtime_hours, failure_weight
PUMPS = [
    ("PMP-001", 3, 13000, 0.50),
    ("PMP-002", 3, 9000, 0.30),
    ("PMP-003", 4, 4000, 0.15),
    ("PMP-004", 5, 1000, 0.05),
]

# Healthy rows favour newer pumps; failures favour older ones (build script).
_HEALTHY_PUMP_P = np.array([0.1, 0.3, 0.3, 0.3])
_FAILURE_MODE_P = np.array([0.5, 0.3, 0.2])
_FAILURE_MODES = ["mechanical", "hydraulic", "stall"]

_ID_CORRUPTIONS = {
    "PMP-001": ["pmp-001", "PMP001", "PMP_001", "PMP - 001"],
    "PMP-002": ["pmp-002", "PMP002", "PMP_002", "PMP - 002"],
    "PMP-003": ["pmp-003", "PMP003", "PMP_003", "PMP - 003"],
    "PMP-004": ["pmp-004", "PMP004", "PMP_004", "PMP - 004"],
}
_RATING_WORDS = {3: "three", 4: "four", 5: "five", 3.0: "three", 4.0: "four", 5.0: "five"}


def _generate_telemetry(
    rng: np.random.Generator, n_samples: int, is_failure: bool
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Return sensor arrays (and per-row failure modes when ``is_failure``)."""
    if not is_failure:
        flow_rate = rng.normal(550, 40, n_samples)
        inlet_pressure = rng.normal(1.5, 0.1, n_samples)
        outlet_pressure = inlet_pressure + rng.normal(2.5, 0.2, n_samples)
        vibration = rng.normal(4.0, 1.0, n_samples)
        temp_k = rng.normal(335, 5, n_samples)
        current = rng.normal(70, 5, n_samples)
        voltage = rng.normal(400, 2, n_samples)
        failure_types = None
    else:
        failure_types = rng.choice(_FAILURE_MODES, size=n_samples, p=_FAILURE_MODE_P)

        flow_rate = np.where(
            failure_types == "hydraulic",
            rng.normal(200, 30, n_samples),
            rng.normal(500, 50, n_samples),
        )
        vibration = np.where(
            failure_types == "mechanical",
            rng.normal(15.0, 2.0, n_samples),
            rng.normal(6.0, 1.5, n_samples),
        )
        temp_k = np.where(
            failure_types == "mechanical",
            rng.normal(375, 8, n_samples),
            rng.normal(345, 5, n_samples),
        )
        current = np.where(
            failure_types == "stall",
            rng.normal(140, 10, n_samples),
            np.where(
                failure_types == "hydraulic",
                rng.normal(25, 5, n_samples),
                rng.normal(75, 8, n_samples),
            ),
        )
        voltage = rng.normal(390, 8, n_samples)
        inlet_pressure = rng.normal(1.5, 0.1, n_samples)
        outlet_pressure = inlet_pressure + rng.normal(2.5, 0.2, n_samples)

    telemetry = {
        "flow_rate": np.maximum(flow_rate, 0),
        "inlet_pressure": np.maximum(inlet_pressure, 0),
        "outlet_pressure": np.maximum(outlet_pressure, 0),
        "vibration_amplitude": np.maximum(vibration, 0),
        "bearing_temperature": np.maximum(temp_k, 290),
        "motor_current": np.maximum(current, 0),
        "motor_voltage": np.maximum(voltage, 0),
    }
    return telemetry, failure_types


def generate(seed: int = 0, n_rows: int = N_ROWS) -> pd.DataFrame:
    """Return a clean, fully labelled telemetry log."""
    rng = np.random.default_rng(seed)

    n_fail = int(n_rows * FAILURE_RATE)
    n_healthy = n_rows - n_fail

    models = [p[0] for p in PUMPS]
    ratings = {p[0]: p[1] for p in PUMPS}
    base_runtime = {p[0]: p[2] for p in PUMPS}
    fail_weights = np.array([p[3] for p in PUMPS], dtype=float)
    fail_weights /= fail_weights.sum()

    healthy_df = pd.DataFrame(
        {
            PUMP_ID_COL: rng.choice(models, size=n_healthy, p=_HEALTHY_PUMP_P),
            TARGET: 0,
        }
    )
    failure_df = pd.DataFrame(
        {
            PUMP_ID_COL: rng.choice(models, size=n_fail, p=fail_weights),
            TARGET: 1,
        }
    )

    df = pd.concat([healthy_df, failure_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df["inspection_rating"] = df[PUMP_ID_COL].map(ratings)
    df["runtime_hours"] = (
        df[PUMP_ID_COL].map(base_runtime) + rng.uniform(0, 10, size=n_rows)
    ).round(1)

    healthy_idx = df[TARGET] == 0
    failure_idx = df[TARGET] == 1

    healthy_telemetry, _ = _generate_telemetry(rng, int(healthy_idx.sum()), is_failure=False)
    failure_telemetry, failure_modes = _generate_telemetry(rng, int(failure_idx.sum()), is_failure=True)

    for col in SENSORS:
        df.loc[healthy_idx, col] = healthy_telemetry[col]
        df.loc[failure_idx, col] = failure_telemetry[col]

    df["failure_mode"] = pd.Series([None] * len(df), dtype=object)
    df.loc[failure_idx, "failure_mode"] = failure_modes
    df[SENSORS] = df[SENSORS].round(3)
    return df


def make_messy(
    df: pd.DataFrame, seed: int = 7, add_duplicates: bool = True
) -> pd.DataFrame:
    """Return a dirtied copy using the homework corruption recipe.

  ``add_duplicates=False`` skips the duplicate-row step so row count stays
  aligned — used when the engine scores a deployment batch position-by-position.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    # 1) Sensor noise — Gaussian jitter, spikes, and dropouts (corrupt script §1).
    noisy_sensor_cols = [
        "vibration_amplitude",
        "bearing_temperature",
        "motor_current",
        "motor_voltage",
    ]
    for col in noisy_sensor_cols:
        std_dev = float(df[col].std())
        out[col] = out[col] + rng.normal(0, std_dev * 0.05, size=len(out))

    for col in noisy_sensor_cols:
        spike_mask = rng.random(len(out)) < 0.01
        spike_magnitude = df[col].max() * rng.uniform(1.5, 3.0, size=len(out))
        out.loc[spike_mask, col] = out.loc[spike_mask, col] + spike_magnitude[spike_mask]

    for col in noisy_sensor_cols:
        dropout_mask = rng.random(len(out)) < 0.01
        out.loc[dropout_mask, col] = np.nan

    for col in noisy_sensor_cols:
        mask = out[col].notna() & (out[col] < 0)
        out.loc[mask, col] = 0

    out[noisy_sensor_cols] = out[noisy_sensor_cols].round(3)

    # 2) String and formatting corruptions (corrupt script §2).
    id_mask = rng.random(len(out)) < 0.05
    for idx in out.index[id_mask]:
        original_id = str(out.at[idx, PUMP_ID_COL])
        if original_id in _ID_CORRUPTIONS:
            out.at[idx, PUMP_ID_COL] = rng.choice(_ID_CORRUPTIONS[original_id])

    out["inspection_rating"] = out["inspection_rating"].astype(object)
    rating_mask = rng.random(len(out)) < 0.05
    for idx in out.index[rating_mask]:
        original_rating = out.at[idx, "inspection_rating"]
        if original_rating in _RATING_WORDS:
            out.at[idx, "inspection_rating"] = _RATING_WORDS[original_rating]

    # 3) Duplicate rows and shuffle (corrupt script §3).
    if add_duplicates:
        n_dupes = int(len(out) * 0.02)
        dupe_idx = rng.choice(out.index.to_numpy(), size=n_dupes, replace=False)
        out = pd.concat([out, out.loc[dupe_idx].copy()], ignore_index=True)
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return out


if __name__ == "__main__":
    clean = generate(seed=42)
    messy = make_messy(clean.drop(columns=["failure_mode"]), seed=42)
    print(clean.shape, "clean rows · failure rate =", round(clean[TARGET].mean(), 3))
    print(messy.shape, "messy rows")
    print(clean["failure_mode"].value_counts(dropna=False))
