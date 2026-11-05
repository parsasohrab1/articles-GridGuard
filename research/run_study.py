"""UCI Electrical Grid Stability (SIMULATED 4-node grid, n=10000): conformal abstention.

usage: python research/run_study.py /path/to/Data_for_UCI_named.csv
Two evaluations over 30 repeats:
  iid    - random 40/30/30 train/calibration/test split
  shift  - train+calibration from the 60 % of grids with the lowest mean reaction time (tau),
           test on the top 20 % (covariate shift in the control parameter)
Class-wise (Mondrian) conformal sets; singleton -> automatic decision, otherwise refer to analyst.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gridguard.conformal import classwise_thresholds, decide, sets
from gridguard.data import load_grid, mean_tau

warnings.filterwarnings("ignore")
ALPHA = 0.05
REPS = 30
UNSTABLE = 1


def evaluate(P, y, th):
    S = sets(P, th)
    dec = decide(S)
    un = y == UNSTABLE
    auto = dec != -1
    argmax = P.argmax(1)
    return {
        "cov_unstable": float(S[un, UNSTABLE].mean()),
        "cov_stable": float(S[~un, 0].mean()),
        "referral_rate": float((~auto).mean()),
        "accuracy_automatic": float((dec[auto] == y[auto]).mean()) if auto.any() else None,
        "unstable_called_stable_automatic": float(((dec == 0) & un).sum() / un.sum()),
        "argmax_accuracy": float((argmax == y).mean()),
        "argmax_unstable_called_stable": float(((argmax == 0) & un).sum() / un.sum()),
    }


def main(path: str):
    X, y, _ = load_grid(path)
    Xn = X.to_numpy()
    tau = mean_tau(X)
    lo, hi = np.quantile(tau, 0.6), np.quantile(tau, 0.8)
    models = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "grad_boosting": lambda: GradientBoostingClassifier(random_state=0),
    }
    out = {
        "note": "SIMULATED grid (UCI 471), not field data; unstable = positive; alpha=0.05; 30 repeats",
        "n": len(y),
        "unstable_share": float(y.mean()),
        "models": {},
    }
    for mname, make in models.items():
        res = {"iid": [], "shift": []}
        for r in range(REPS):
            idx = np.arange(len(y))
            tr, rest = train_test_split(idx, test_size=0.6, random_state=r, stratify=y)
            ca, te = train_test_split(rest, test_size=0.5, random_state=r, stratify=y[rest])
            m = make().fit(Xn[tr], y[tr])
            res["iid"].append(
                evaluate(
                    m.predict_proba(Xn[te]),
                    y[te],
                    classwise_thresholds(m.predict_proba(Xn[ca]), y[ca], ALPHA),
                )
            )
            ind = np.where(tau <= lo)[0]
            sh = np.where(tau >= hi)[0]
            tr2, ca2 = train_test_split(ind, test_size=0.4, random_state=r, stratify=y[ind])
            m2 = make().fit(Xn[tr2], y[tr2])
            res["shift"].append(
                evaluate(
                    m2.predict_proba(Xn[sh]),
                    y[sh],
                    classwise_thresholds(m2.predict_proba(Xn[ca2]), y[ca2], ALPHA),
                )
            )
        out["models"][mname] = {
            k: {m: float(np.mean([d[m] for d in v if d[m] is not None])) for m in v[0]}
            for k, v in res.items()
        }
        print(mname)
        for k, v in out["models"][mname].items():
            print("  ", k, {a: round(b, 3) for a, b in v.items()})
    out["shift_definition"] = {
        "train_cal_tau_max": float(lo),
        "test_tau_min": float(hi),
        "unstable_share_in_train": float(y[tau <= lo].mean()),
        "unstable_share_in_test": float(y[tau >= hi].mean()),
    }
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
