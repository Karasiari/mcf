from __future__ import annotations

from typing import Iterable, Optional, Tuple

import networkx as nx
import numpy as np

from src.solvers.throughput.solvers import CvxpyThroughputSolver, HighsThroughputSolver

from tqdm import tqdm


def find_mat_for_throughput(
    graph: nx.Graph,
    best_mat: np.ndarray,
    worst_mat: np.ndarray,
    target: float,
    rtol: float = 1e-4,
    max_iter: int = 30,
    solver: Optional[object] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, float]:
    s = solver or CvxpyThroughputSolver(graph)
    lo, hi = 0.0, 1.0
    tol = rtol * target
    traffic_mat = best_mat
    alpha, gamma = 1.0, 0.0
    for _ in range(max_iter):
        alpha = 0.5 * (lo + hi)
        traffic_mat = alpha * best_mat + (1 - alpha) * worst_mat
        sol = s.solve(traffic_mat)
        gamma = float(sol.gamma)
        if abs(gamma - target) < tol:
            break
        if gamma < target:
            lo = alpha
        else:
            hi = alpha
    return traffic_mat, alpha, gamma


def _choose_two_offdiag(rng: np.random.Generator, n: int) -> tuple[tuple[int, int], tuple[int, int]]:
    i1, j1 = tuple(rng.integers(0, n, size=2))
    while i1 == j1:
        i1, j1 = tuple(rng.integers(0, n, size=2))
    i2, j2 = tuple(rng.integers(0, n, size=2))
    while i2 == j2 or (i2 == i1 and j2 == j1):
        i2, j2 = tuple(rng.integers(0, n, size=2))
    return (i1, j1), (i2, j2)


def sample_tm_random_walk(
    graph: nx.Graph,
    target: float,
    width: float,
    eps: float,
    steps: int,
    start_mat: np.ndarray,
    solver: Optional[object] = None,
    seed: Optional[int] = None,
    max_tries_per_step: int = 1000,
) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    s = solver or HighsThroughputSolver(graph)
    low, high = target - width / 2, target + width / 2
    n = start_mat.shape[0]
    x = start_mat.copy()
    samples: list[np.ndarray] = []
    for _ in tqdm(range(steps)):
        stepped = False
        for _ in range(max_tries_per_step):
            tries = 0
            for tries in range(int(1e6)):
                if tries and not tries % int(1e5):
                    print("cant find large enough offdiag. eps too high?")
                (a, b), (c, d) = _choose_two_offdiag(rng, n)
                if x[c, d] >= eps:
                   break 
            x[a, b] += eps
            x[c, d] -= eps
            gamma = float(s.solve(x).gamma)
            if low <= gamma <= high:
                samples.append(x.copy())
                stepped = True
                break
            x[a, b] -= eps
            x[c, d] += eps
        if not stepped:
            print("warn: max tries hit")
    return samples


