"""Part 1 — MISSING code: predict() was never implemented (the method is absent)."""
import numpy as np


class PumpSolution:
    def clean(self, df):
        out = df.copy()
        num = out.select_dtypes("number").columns
        out[num] = out[num].fillna(out[num].median())
        return out.drop_duplicates().reset_index(drop=True)

    # predict() intentionally missing — the student left this task blank.


def get_solution():
    return PumpSolution()
