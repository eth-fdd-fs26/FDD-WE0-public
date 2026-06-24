"""Synthetic basin-temperature data — Part 2 ("Predict the basin temperature").

The reactor's main basin must not overheat. We generate a small regression
problem: from a handful of plant sensors, predict the basin temperature (°C).
The relationship is mostly smooth and learnable, so a tiny MLP reaches a low
MAE — and a broken architecture (the Part-2 bugs) visibly fails to.

Features (in column order ``FEATURES``):
  * reactor_load   — normalised thermal load (0.2 .. 1.0)
  * coolant_flow   — normalised coolant flow (0.4 .. 1.2)  (more flow → cooler)
  * ambient_temp   — outside temperature (°C)
  * pump_vibration — coolant-pump vibration (mm/s)  (mild heating proxy)
  * hour           — hour of day (0..23) → diurnal swing
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["reactor_load", "coolant_flow", "ambient_temp", "pump_vibration", "hour"]
TARGET = "basin_temp"

# Basin overheats above this; the early-warning alarm should fire before it.
OVERHEAT_C = 85.0


def generate(seed: int = 0, n: int = 4000) -> pd.DataFrame:
    """Return a labelled frame with FEATURES + the ``basin_temp`` target."""
    rng = np.random.default_rng(seed)
    load = rng.uniform(0.2, 1.0, n)
    flow = rng.uniform(0.4, 1.2, n)
    ambient = rng.normal(22.0, 5.0, n)
    vib = np.abs(rng.normal(2.6, 0.7, n))
    hour = rng.integers(0, 24, n).astype(float)

    # Smooth "physics": load heats, flow cools (nonlinearly), ambient couples in,
    # vibration adds a little friction heat, plus a daily sine swing + noise.
    temp = (
        46.0
        + 42.0 * load
        - 18.0 * (flow - 0.4)
        + 0.45 * (ambient - 22.0)
        + 1.8 * (vib - 2.6)
        + 6.0 * load / flow                       # hot + starved of coolant = worst
        + 3.0 * np.sin((hour - 6) / 24 * 2 * np.pi)
        + rng.normal(0, 1.2, n)
    )

    df = pd.DataFrame({
        "reactor_load": load,
        "coolant_flow": flow,
        "ambient_temp": ambient,
        "pump_vibration": vib,
        "hour": hour,
        TARGET: temp,
    })
    return df


def Xy(df: pd.DataFrame):
    """Split a frame into a float32 feature matrix and a target vector."""
    X = df[FEATURES].to_numpy(dtype="float32")
    y = df[TARGET].to_numpy(dtype="float32")
    return X, y


if __name__ == "__main__":
    d = generate()
    print(d.describe().round(2))
    print("overheat fraction:", round((d[TARGET] > OVERHEAT_C).mean(), 3))
