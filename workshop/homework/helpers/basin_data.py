"""Synthetic data for the reactor basin-temperature notebook (HW3).

The story: the plant's main-basin temperature sensors are failing under radiation,
so we predict the temperature from a *backup* system that fuses two very different
signals:

  * an array of **secondary sensors** sampled over a short window — a 1-D
    multi-channel time series (coolant flow, pump rpm, ambient, neutron flux);
  * a **thermal camera**, here represented as a few small frames whose hot-spot
    brightness tracks the real temperature (in the notebook these go through a
    frozen vision encoder, exactly as if the features had been pre-extracted).

We don't have a real reactor, so this module fabricates a deterministic dataset
that *is learnable from both modalities at once*: the target depends on a temporal
feature only the sensor series exposes (the recent neutron-flux trend) AND on the
thermal hot-spot — so a model has to use both branches to do well.

    import basin_data
    b = basin_data.load_basin()
    b["X_sensor"]   # (N, 4, 64)        float32  — 4 channels, 64 timesteps
    b["X_thermal"]  # (N, 3, 16, 16, 1) float32  — 3 frames, 16x16, 1 channel (HWC)
    b["y"]          # (N,)              float32  — basin temperature, STANDARDIZED
    b["y_mean"], b["y_std"]             # °C scale, to map predictions back
    b["overheat_c"]                     # the °C line that means "danger"

Everything is seeded, so the data is identical every run and the notebook is
reproducible offline.
"""
import numpy as np

# physical-ish constants used only to make the story concrete (and the plots nice)
SENSOR_NAMES = ["coolant flow", "pump rpm", "ambient", "neutron flux"]
N_CHANNELS = 4
N_STEPS = 64
N_FRAMES = 3
IMG = 16
OVERHEAT_C = 340.0          # above this the manager must act


def load_basin(n=3000, seed=0):
    """Return the basin bundle (see module docstring). Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, N_STEPS, dtype=np.float32)        # within-window time

    # ----- secondary sensor time series, (N, 4, 64) -----------------------
    # Each sample gets its own random phases/levels; the channels are smooth
    # waves plus noise. The one that matters for temperature is the neutron flux
    # (channel 3): its *upward trend over the recent window* drives heat-up.
    flow_level = rng.uniform(0.3, 0.9, size=n).astype(np.float32)
    pump_phase = rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)
    ambient = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    flux_slope = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)   # the hidden driver

    X = np.empty((n, N_CHANNELS, N_STEPS), dtype=np.float32)
    X[:, 0] = flow_level[:, None] + 0.05 * np.sin(6 * t)[None, :]            # coolant flow
    X[:, 1] = np.sin(8 * t[None, :] + pump_phase[:, None])                   # pump rpm
    X[:, 2] = ambient[:, None] + 0.1 * t[None, :]                            # ambient drift
    X[:, 3] = (flux_slope[:, None] * (t - 0.5)[None, :] * 2.0                # neutron flux ramp
               + 0.3 * np.sin(12 * t)[None, :])
    X += 0.05 * rng.standard_normal(X.shape).astype(np.float32)             # sensor noise

    # ----- thermal camera frames, (N, 3, 16, 16, 1) — HWC, 1 channel ------
    # A warm hot-spot in the centre whose brightness encodes a second, independent
    # contribution to the temperature. Stored channels-LAST (like NumPy/PIL) so the
    # notebook has to permute the axes before the (channels-first) vision encoder.
    hotspot = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    bump = np.exp(-((xx - IMG / 2) ** 2 + (yy - IMG / 2) ** 2) / 18.0).astype(np.float32)
    base = 0.2 + 0.1 * rng.standard_normal((n, 1, 1)).astype(np.float32)
    frames = (base[:, None] + hotspot[:, None, None, None] * bump[None, None]
              + 0.03 * rng.standard_normal((n, N_FRAMES, IMG, IMG)).astype(np.float32))
    X_thermal = frames[..., None].astype(np.float32)            # add channel axis -> HWC

    # ----- the target: needs BOTH modalities -------------------------------
    # flux trend (only the time series shows it) + hot-spot (only the camera shows it)
    flux_recent = X[:, 3, N_STEPS // 2:].mean(axis=1)          # recent neutron-flux level
    temp_c = (300.0
              + 22.0 * flux_recent
              + 16.0 * hotspot
              + 6.0 * (flow_level - 0.6)                       # mild cooling effect
              + 2.0 * rng.standard_normal(n).astype(np.float32))
    temp_c = temp_c.astype(np.float32)

    y_mean = float(temp_c.mean())
    y_std = float(temp_c.std())
    y = ((temp_c - y_mean) / y_std).astype(np.float32)         # standardized target

    return {
        "X_sensor": X,
        "X_thermal": X_thermal,
        "y": y,
        "temp_c": temp_c,                 # raw °C, handy for the final plot
        "y_mean": y_mean,
        "y_std": y_std,
        "overheat_c": OVERHEAT_C,
        "sensor_names": SENSOR_NAMES,
        "n_channels": N_CHANNELS,
        "n_steps": N_STEPS,
        "n_frames": N_FRAMES,
        "img": IMG,
    }


if __name__ == "__main__":
    b = load_basin()
    print("X_sensor :", b["X_sensor"].shape, b["X_sensor"].dtype)
    print("X_thermal:", b["X_thermal"].shape, b["X_thermal"].dtype)
    print("y        :", b["y"].shape, "(standardized) ·  °C mean/std:",
          round(b["y_mean"], 1), "/", round(b["y_std"], 1))
    print("overheat at", b["overheat_c"], "°C ·  share over:",
          round(float((b["temp_c"] > b["overheat_c"]).mean()), 3))
