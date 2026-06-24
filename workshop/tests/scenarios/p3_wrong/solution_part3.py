"""Part 3 — WRONG code: repair and clustering are correct (inherited from the
reference), but route() re-clusters the new arrivals from scratch instead of
reusing the fitted machines. The ids look valid, yet they don't match the
"nearest configured machine" — the routing oracle catches it.
"""
import os
import sys

import numpy as np
from sklearn.cluster import KMeans

# make the reference solution importable, then inherit its (correct) fill_gaps +
# build_machine_plan and override only route().
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from workshop.backend.reference.solution_part3 import WasteSolution as _Ref  # noqa: E402
import hw3_data  # noqa: E402  (path was injected by the reference module)


class WasteSolution(_Ref):
    def route(self, plan, new_df):
        feats = new_df[hw3_data.PHYSICAL_FEATURES + hw3_data.CATEGORICAL]
        Xn = np.asarray(plan["prep"].transform(feats))
        # WRONG: fit a brand-new KMeans on the arrivals instead of plan["km"].predict
        return KMeans(n_clusters=plan["k"], n_init=10, random_state=1).fit_predict(Xn)


def get_solution():
    return WasteSolution()
