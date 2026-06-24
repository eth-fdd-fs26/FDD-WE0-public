"""Part 3 — MISSING code: fill_gaps() was never implemented, so it returns
tomorrow's log untouched and the missing values are still there."""


class WasteSolution:
    def fill_gaps(self, history, gappy):
        return gappy                         # the gaps are never filled

    def build_machine_plan(self, df):
        raise RuntimeError("unreachable — repair fails first")

    def route(self, plan, new_df):
        raise RuntimeError("unreachable — repair fails first")


def get_solution():
    return WasteSolution()
