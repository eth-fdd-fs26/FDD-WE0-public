"""Part 3 — BAD results: fill_gaps() runs and leaves no NaNs, but it fills every
missing quantity with a single flat mean and every missing critical-flag with 0.
No NaNs remain, yet the imputed values are worthless → quantity R² collapses."""


class WasteSolution:
    def fill_gaps(self, history, gappy):
        out = gappy.copy()
        out["quantity_kg"] = out["quantity_kg"].fillna(history["quantity_kg"].mean())
        out["critical"] = out["critical"].fillna(0).astype(int)
        return out

    def build_machine_plan(self, df):
        raise RuntimeError("unreachable — repair fails first")

    def route(self, plan, new_df):
        raise RuntimeError("unreachable — repair fails first")


def get_solution():
    return WasteSolution()
