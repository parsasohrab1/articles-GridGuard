from __future__ import annotations

import numpy as np


def _q(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if k <= n else float("inf")


def classwise_thresholds(P: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    th = np.full(P.shape[1], np.inf)
    for c in range(P.shape[1]):
        m = y == c
        if m.sum():
            th[c] = _q(1 - P[m, c], alpha)
    return th


def sets(P: np.ndarray, th: np.ndarray) -> np.ndarray:
    return (1 - P) <= th


def decide(S: np.ndarray) -> np.ndarray:
    """-1 = refer (empty or both labels), otherwise the single label in the set."""
    out = np.full(len(S), -1)
    single = S.sum(1) == 1
    out[single] = S[single].argmax(1)
    return out
