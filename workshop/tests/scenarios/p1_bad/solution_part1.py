"""Part 1 — BAD results: predict() runs and returns valid 0/1, but always says
'safe' (the majority class). It passes the contract but catches zero failures,
so recall is 0 → the pump seizes."""
import numpy as np


class PumpSolution:
    def clean(self, df):
        out = df.copy()
        num = out.select_dtypes("number").columns
        out[num] = out[num].fillna(out[num].median())
        return out.drop_duplicates().reset_index(drop=True)

    def predict(self, df):
        return np.zeros(len(df), dtype=int)   # "everything is fine" — always


def get_solution():
    return PumpSolution()
