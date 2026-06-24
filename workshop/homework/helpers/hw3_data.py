"""Data helper for the HW3 notebook ("Configuring the Night-Shift Waste Machines").

Provides one synthetic table of radioactive-waste batches that the clustering
exercise works on. Import it from the notebook:

    import hw3_data
    waste_df = hw3_data.load_data()

Under the hood it samples each batch from one of a few *hidden* waste archetypes
(distinct physical profiles), so a genuine cluster structure exists and the
silhouette / elbow search has a clear optimum. The hidden archetype label is NOT
exposed — the exercise is unsupervised. On top of the meaningful physical
features it bolts on:
  * pure-noise / identifier columns (to be removed in feature selection), and
  * a redundant column (`radioactivity_Ci`) that is just `radioactivity_Bq` in
    different units (to be caught by the correlation step).

Everything is seeded, so the dataset is identical every run. The table is cached
to data/hw3_waste.csv (read back if present, regenerated otherwise).

Resolution order:
  1. data/hw3_waste.csv   (pre-baked — first hit wins)
  2. generate from scratch (seeded) and cache it there
"""
import os

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- column groups
# Pure bookkeeping — no physical signal. Feature selection should drop these.
ID_COLS = ["batch_id"]
NOISE_COLS = ["storage_shelf", "barcode"]

# The physical features that actually describe the waste (these drive clusters).
PHYSICAL_FEATURES = [
    "radioactivity_Bq",   # becquerel — spans many orders of magnitude
    "half_life_years",    # spans many orders of magnitude
    "heat_output_W",      # decay heat per batch
    "density_kg_m3",
    "corrosiveness",      # 0..1 index
    "particle_size_mm",
]

# Same information as radioactivity_Bq, only in curies — correlation ~ 1.0.
REDUNDANT_FEATURES = ["radioactivity_Ci"]

# Categorical physical feature (one-hot it during preprocessing).
CATEGORICAL = ["chemical_class"]

# Heavily skewed features that benefit from a log transform before scaling.
LOG_FEATURES = ["radioactivity_Bq", "half_life_years", "heat_output_W"]

# Per-batch mass — used for the final "how much does each machine process" report,
# NOT for the clustering geometry.
QUANTITY_COL = "quantity_kg"

CHEM_CLASSES = ["organic", "metallic", "ceramic", "aqueous"]
SHELVES = [f"S{n:02d}" for n in range(1, 25)]

BQ_PER_CI = 3.7e10  # 1 curie = 3.7e10 becquerel

# ----------------------------------------------------------------- hidden archetypes
# Five distinct waste "kinds". Each row in the table is sampled from one of them.
# Values are (mean, lognormal-sigma or spread) per feature; chem_probs biases the
# categorical column so the structure is real but not perfectly separable.
ARCHETYPES = [
    dict(name="spent_fuel",   weight=0.22,
         radioactivity_Bq=4e12, half_life_years=3e4,  heat_output_W=900.0,
         density=10500, corrosiveness=0.25, particle_size=8.0,
         chem_probs=[0.05, 0.70, 0.20, 0.05], qty_mean=180, critical_p=0.95),
    dict(name="medical",      weight=0.20,
         radioactivity_Bq=2e8,  half_life_years=0.02, heat_output_W=3.0,
         density=1100,  corrosiveness=0.35, particle_size=1.5,
         chem_probs=[0.65, 0.05, 0.05, 0.25], qty_mean=12, critical_p=0.15),
    dict(name="contaminated_metal", weight=0.24,
         radioactivity_Bq=5e9,  half_life_years=30.0, heat_output_W=25.0,
         density=7800,  corrosiveness=0.80, particle_size=20.0,
         chem_probs=[0.05, 0.80, 0.10, 0.05], qty_mean=320, critical_p=0.80),
    dict(name="process_sludge", weight=0.20,
         radioactivity_Bq=8e7,  half_life_years=12.0, heat_output_W=6.0,
         density=1400,  corrosiveness=0.55, particle_size=0.4,
         chem_probs=[0.30, 0.05, 0.05, 0.60], qty_mean=540, critical_p=0.30),
    dict(name="lab_glassware", weight=0.14,
         radioactivity_Bq=1e6,  half_life_years=0.3,  heat_output_W=0.5,
         density=2500,  corrosiveness=0.20, particle_size=12.0,
         chem_probs=[0.10, 0.05, 0.75, 0.10], qty_mean=40, critical_p=0.05),
]

# The "right answer" the silhouette / elbow search should recover.
N_TRUE_CLUSTERS = len(ARCHETYPES)

# Extra columns used by the later parts (routing / forecasting).
DAY_COL = "day"
CRITICAL_COL = "critical"   # 1 = critical / special-handling waste, 0 = routine


def _one_batch(rng, a, idx, day=None, with_critical=False):
    """Sample a single waste batch from archetype `a` (lognormal scatter)."""
    bq = a["radioactivity_Bq"] * rng.lognormal(0.0, 0.35)
    row = {
        "batch_id": f"W-{idx:05d}",
        "storage_shelf": rng.choice(SHELVES),                       # pure noise
        "barcode": int(rng.integers(10_000_000, 99_999_999)),       # pure noise
        "radioactivity_Bq": bq,
        "radioactivity_Ci": bq / BQ_PER_CI * rng.normal(1.0, 0.002),  # redundant
        "half_life_years": a["half_life_years"] * rng.lognormal(0.0, 0.40),
        "heat_output_W": a["heat_output_W"] * rng.lognormal(0.0, 0.35),
        "density_kg_m3": a["density"] * rng.normal(1.0, 0.06),
        "corrosiveness": float(np.clip(a["corrosiveness"] + rng.normal(0, 0.07), 0.0, 1.0)),
        "particle_size_mm": max(0.05, a["particle_size"] * rng.lognormal(0.0, 0.30)),
        "chemical_class": rng.choice(CHEM_CLASSES, p=a["chem_probs"]),
        "quantity_kg": max(1.0, a["qty_mean"] * rng.lognormal(0.0, 0.30)),
    }
    if day is not None:
        row[DAY_COL] = int(day)
    if with_critical:
        # Critical status is mostly driven by the archetype but genuinely noisy,
        # so predicting it from the physical features is a real classification task.
        row[CRITICAL_COL] = int(rng.random() < a["critical_p"])
    return row


def _column_order(with_day=False, with_critical=False):
    cols = []
    if with_day:
        cols.append(DAY_COL)
    cols += (ID_COLS + NOISE_COLS + PHYSICAL_FEATURES[:1] + REDUNDANT_FEATURES
             + PHYSICAL_FEATURES[1:] + CATEGORICAL + [QUANTITY_COL])
    if with_critical:
        cols.append(CRITICAL_COL)
    return cols


def _sample(n_rows, seed, days=None, with_critical=False):
    """Sample `n_rows` batches. If `days` is given, spread them across day labels."""
    rng = np.random.default_rng(seed)
    weights = np.array([a["weight"] for a in ARCHETYPES])
    weights = weights / weights.sum()
    assign = rng.choice(len(ARCHETYPES), size=n_rows, p=weights)
    day_labels = rng.integers(1, days + 1, size=n_rows) if days else [None] * n_rows

    rows = [_one_batch(rng, ARCHETYPES[a_idx], i, day=d, with_critical=with_critical)
            for i, (a_idx, d) in enumerate(zip(assign, day_labels))]
    df = pd.DataFrame(rows)[_column_order(with_day=days is not None, with_critical=with_critical)]
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _generate(n_rows=520, seed=0):
    return _sample(n_rows, seed)


def _cache_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, os.pardir, "data", "hw3_waste.csv")


def load_data(verbose=True):
    """Return the waste-batch DataFrame (cached to data/hw3_waste.csv)."""
    path = _cache_path()
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = _generate()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    if verbose:
        print(f"Loaded {len(df)} waste batches · {df.shape[1]} columns ✅")
    return df


def load_incoming(n=12, seed=42, verbose=True):
    """A fresh batch of *new, unlabelled* waste just delivered (same raw schema).

    Used in Part 6 to route new arrivals into the machines we already fitted.
    """
    df = _sample(n, seed)
    if verbose:
        print(f"{len(df)} new waste batches just arrived 🚚")
    return df


def load_history(n_days=60, per_day=70, seed=7, verbose=True):
    """A COMPLETE batch log collected over many days (with `day` + `critical`).

    This is the training data for the predictors in Part 7.
    """
    df = _sample(n_days * per_day, seed, days=n_days, with_critical=True)
    if verbose:
        print(f"Historical log: {len(df)} batches across {n_days} days ✅")
    return df


def load_next_day(n=140, seed=99, missing_frac=0.25, verbose=True):
    """Tomorrow's intake (same schema as the history) but with GAPS:

    a fraction of `quantity_kg` (amount) and `critical` (presence) cells are
    blanked to NaN. Part 7 trains models on the history to fill them back in.
    """
    df = _sample(n, seed, days=1, with_critical=True)
    df[DAY_COL] = df[DAY_COL] + 1000  # mark it as a separate, future day
    rng = np.random.default_rng(seed + 1)
    for col in (QUANTITY_COL, CRITICAL_COL):
        mask = rng.random(len(df)) < missing_frac
        df.loc[mask, col] = np.nan
    if CRITICAL_COL in df:
        df[CRITICAL_COL] = df[CRITICAL_COL].astype("float")  # allow NaN
    if verbose:
        miss_q = int(df[QUANTITY_COL].isna().sum())
        miss_c = int(df[CRITICAL_COL].isna().sum())
        print(f"Tomorrow's intake: {len(df)} batches · "
              f"{miss_q} missing quantities · {miss_c} missing critical-flags ⚠️")
    return df


if __name__ == "__main__":
    # Force a fresh regeneration and report the planted structure.
    df = _generate()
    os.makedirs(os.path.dirname(_cache_path()), exist_ok=True)
    df.to_csv(_cache_path(), index=False)
    print(f"Wrote {_cache_path()} · {df.shape}")
    print(f"Planted clusters (hidden): {N_TRUE_CLUSTERS}")
    print(df.head())
