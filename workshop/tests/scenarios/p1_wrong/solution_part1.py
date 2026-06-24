"""Part 1 — WRONG code: predict() returns probabilities (floats), not 0/1 labels.

A classic mix-up: returning ``predict_proba`` output instead of class labels.
The values are the right length but aren't 0/1, so the contract check fails.
"""
import numpy as np


class PumpSolution:
    def clean(self, df):
        out = df.copy()
        num = out.select_dtypes("number").columns
        out[num] = out[num].fillna(out[num].median())
        return out.drop_duplicates().reset_index(drop=True)

    def predict(self, df):
        # continuous risk scores in [0, 1] — NOT hard 0/1 labels
        v = df["vibration"].fillna(2.5).to_numpy()
        return np.clip((v - 2.0) / 10.0, 0.0, 1.0)


def get_solution():
    return PumpSolution()
