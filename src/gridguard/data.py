from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["tau1", "tau2", "tau3", "tau4", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]


def load_grid(path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """UCI 'Electrical Grid Stability Simulated Data' (simulation of a 4-node star grid).

    Returns features, y (1 = unstable) and the continuous stability margin ``stab``
    (max real part of the characteristic-equation roots; > 0 means unstable).
    p1 is dropped: it equals -(p2 + p3 + p4).
    """
    d = pd.read_csv(path)
    y = (d["stabf"] == "unstable").astype(int).to_numpy()
    return d[FEATURES], y, d["stab"].to_numpy()


def mean_tau(X: pd.DataFrame) -> np.ndarray:
    return X[["tau1", "tau2", "tau3", "tau4"]].mean(axis=1).to_numpy()
