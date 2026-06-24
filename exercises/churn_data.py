"""Placeholder churn data loader for the Block 3 PyTorch notebook.

The *previous* notebook (which introduced PyTorch on the customer-churn dataset)
shipped its own data loader; we don't have access to it yet, so this module is a
deterministic stand-in that returns the same shape of thing that loader would:

    import churn_data
    bundle = churn_data.load_churn()
    bundle["X_num"]   # (N, n_numeric)  float32   — scaled numeric features
    bundle["X_cat"]   # (N, n_cat)      int64     — category indices per column
    bundle["y"]       # (N,)            int64     — churn label (0/1)
    bundle["numeric_names"], bundle["categorical_names"], bundle["cat_cardinalities"]

It mimics the Telco "customer churn" schema (scikit-learn/churn-prediction on the
Hugging Face Hub): a handful of numeric account fields plus several low-cardinality
categoricals, with a churn flag that depends on them. Everything is seeded, so the
data is identical every run and the notebook is reproducible offline. When the real
upstream loader lands, swap `load_churn` to call it and keep the same return keys.
"""
import numpy as np

# ---- schema -----------------------------------------------------------------
NUMERIC_NAMES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = {
    "gender":          ["Female", "Male"],
    "Contract":        ["Month-to-month", "One year", "Two year"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaymentMethod":   ["Electronic check", "Mailed check",
                        "Bank transfer", "Credit card"],
}
CATEGORICAL_NAMES = list(CATEGORICAL)
CAT_CARDINALITIES = [len(v) for v in CATEGORICAL.values()]


def load_churn(n=8000, seed=0):
    """Return a churn bundle (see module docstring). Deterministic given `seed`."""
    rng = np.random.default_rng(seed)

    # --- raw numeric fields -------------------------------------------------
    tenure = rng.integers(0, 72, size=n).astype(np.float32)              # months
    monthly = rng.uniform(18.0, 120.0, size=n).astype(np.float32)        # $/month
    # total roughly = tenure * monthly, with noise (and the classic 0-tenure quirk)
    total = (tenure * monthly * rng.uniform(0.8, 1.05, size=n)).astype(np.float32)

    # --- categoricals as integer indices ------------------------------------
    cat_cols = []
    cat_probs = {
        "gender":          [0.50, 0.50],
        "Contract":        [0.55, 0.25, 0.20],
        "InternetService": [0.35, 0.45, 0.20],
        "PaymentMethod":   [0.35, 0.22, 0.22, 0.21],
    }
    cat_idx = {}
    for name, levels in CATEGORICAL.items():
        idx = rng.choice(len(levels), size=n, p=cat_probs[name])
        cat_idx[name] = idx
        cat_cols.append(idx)
    X_cat = np.stack(cat_cols, axis=1).astype(np.int64)

    # --- a churn rule the model can actually learn --------------------------
    # Short tenure, high monthly charge, month-to-month contract, fiber + e-check
    # all push churn up. Logit built from those, then sampled.
    # The categoricals carry most of the signal here — so a model that scrambles
    # them (the embedding-reshape bug in torch_lab) really cannot learn.
    z = (
        -0.7
        - 0.020 * (tenure - 18)
        + 0.008 * (monthly - 65)
        + 1.8 * (cat_idx["Contract"] == 0)            # month-to-month churns hard
        - 1.6 * (cat_idx["Contract"] == 2)            # two-year locks people in
        + 1.3 * (cat_idx["InternetService"] == 1)     # fiber optic churns more
        - 0.9 * (cat_idx["InternetService"] == 2)     # no internet rarely churns
        + 1.1 * (cat_idx["PaymentMethod"] == 0)       # electronic check
    )
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p).astype(np.int64)

    # --- scale numeric features (standardize) -------------------------------
    X_num = np.stack([tenure, monthly, total], axis=1).astype(np.float32)
    mean = X_num.mean(axis=0, keepdims=True)
    std = X_num.std(axis=0, keepdims=True) + 1e-8
    X_num = (X_num - mean) / std

    return {
        "X_num": X_num,
        "X_cat": X_cat,
        "y": y,
        "numeric_names": NUMERIC_NAMES,
        "categorical_names": CATEGORICAL_NAMES,
        "cat_cardinalities": CAT_CARDINALITIES,
    }


if __name__ == "__main__":
    b = load_churn()
    print("X_num:", b["X_num"].shape, "· X_cat:", b["X_cat"].shape,
          "· churn rate:", round(float(b["y"].mean()), 3))
