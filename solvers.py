from __future__ import annotations

from typing import Optional, Tuple

import networkx as nx
import numpy as np

import cvxpy as cp

from src.data.types import FLOAT
from src.solvers.edge_based.solution import Solution
from src.solvers.edge_based.simplex import get_data_for_lp
from src.utils import (
    calculate_laplacian_from_weights_matrix,
    get_incidence_matrix,
    get_var_value,
    get_weights,
)


class CvxpyThroughputSolver:
    def __init__(self, graph: nx.Graph, solver: str = "SCS", solver_opts: Optional[dict] = None):
        g = nx.DiGraph(graph)
        self.n, self.m = g.number_of_nodes(), g.number_of_edges()
        self.inc = get_incidence_matrix(g)
        self.bandwidth = get_weights(g, "bandwidth")
        self.L = cp.Parameter((self.n, self.n))
        self.F = cp.Variable((self.m, self.n))
        self.gamma = cp.Variable()
        self.prob = cp.Problem(
            cp.Maximize(self.gamma),
            [
                cp.sum(self.F, axis=1) <= self.bandwidth,
                self.inc @ self.F == -self.gamma * self.L.T,
                self.F >= 0,
                self.gamma >= 0,
            ],
        )
        self.solver = solver
        self.solver_opts = {} if solver_opts is None else solver_opts

    def solve(self, traffic_mat: np.ndarray, warm_start: bool = True) -> Solution:
        self.L.value = calculate_laplacian_from_weights_matrix(traffic_mat, "out")
        self.prob.solve(solver=getattr(cp, self.solver), warm_start=warm_start, **self.solver_opts)
        return Solution(problem=self.prob, flow=get_var_value(self.F), gamma=get_var_value(self.gamma))


class HighsThroughputSolver:
    def __init__(self, graph: nx.Graph):
        try:
            from highspy import Highs
        except Exception as e:
            raise ImportError("highspy is required for HighsThroughputSolver") from e
        self.Highs = Highs
        g = nx.DiGraph(graph)
        self.n, self.m = g.number_of_nodes(), g.number_of_edges()
        self.inc = get_incidence_matrix(g).astype(FLOAT)
        self.bandwidth = get_weights(g, "bandwidth").astype(FLOAT)
        self.R = self.n * self.n + self.m
        self.N = self.m * self.n + 1
        self.h = self.Highs()
        self._built = False
        self._fallback = False

    def _build_column_sparse(self, L_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Aindex: list[int] = []
        Avalue: list[FLOAT] = []
        Astart = [0]
        for k in range(self.n):
            eq_row_start = k * self.n
            for e in range(self.m):
                inc_col = self.inc[:, e]
                nz_eq = np.nonzero(inc_col)[0]
                Aindex.extend((eq_row_start + nz_eq).tolist())
                Avalue.extend(inc_col[nz_eq].tolist())
                Aindex.append(self.n * self.n + e)
                Avalue.append(1.0)
                Astart.append(len(Aindex))
        nz = np.nonzero(L_flat)[0]
        Aindex.extend(nz.tolist())
        Avalue.extend((-L_flat[nz]).tolist())
        Astart.append(len(Aindex))
        return np.array(Astart, dtype=np.int64), np.array(Aindex, dtype=np.int32), np.array(Avalue, dtype=FLOAT)

    def _row_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        rl = np.zeros(self.n * self.n + self.m, dtype=FLOAT)
        ru = np.zeros(self.n * self.n + self.m, dtype=FLOAT)
        rl[self.n * self.n :] = -np.inf
        ru[self.n * self.n :] = self.bandwidth
        return rl, ru

    def _build(self, L_flat: np.ndarray):
        c = np.zeros(self.N, dtype=FLOAT)
        c[-1] = -1.0
        lb = np.zeros(self.N, dtype=FLOAT)
        ub = np.full(self.N, np.inf, dtype=FLOAT)
        Astart, Aindex, Avalue = self._build_column_sparse(L_flat)
        rl, ru = self._row_bounds()
        try:
            # Try passing the full model in one call if available
            nnz = len(Avalue)
            self.h.passModel(self.N, self.R, nnz, c, lb, ub, rl, ru, Astart, Aindex, Avalue)
            self._built = True
        except TypeError:
            # Fallback: add columns and then rows if API differs
            try:
                self.h.addCols(self.N, c, lb, ub, Astart, Aindex, Avalue)
                self.h.addRows(self.R, rl, ru)
                self._built = True
            except Exception:
                self._fallback = True

    def _update_gamma_column(self, L_flat: np.ndarray):
        for r in range(self.n * self.n):
            v = L_flat[r]
            self.h.changeCoeff(r, self.N - 1, float(v))

    def solve(self, traffic_mat: np.ndarray) -> Solution:
        if self._fallback:
            return self._solve_via_scipy(traffic_mat)
        L_flat = calculate_laplacian_from_weights_matrix(traffic_mat, "out").flatten().astype(FLOAT)
        if not self._built:
            self._build(L_flat)
        if self._fallback:
            return self._solve_via_scipy(traffic_mat)
        else:
            self._update_gamma_column(L_flat)
            self.h.setOptionValue("solver", "simplex")
            self.h.setOptionValue("simplex_strategy", "dual")
            self.h.run()
            sol = self.h.getSolution()
            x = np.array(sol.col_value, dtype=FLOAT)
            flow_vec = x[: self.m * self.n]
            flow = flow_vec.reshape(self.n, self.m).T
            gamma = float(x[-1])
            return Solution(problem=None, flow=flow, gamma=gamma)

    def _solve_via_scipy(self, traffic_mat: np.ndarray) -> Solution:
        from scipy.optimize import linprog
        L = calculate_laplacian_from_weights_matrix(traffic_mat, "out").astype(FLOAT)
        K_blocks = [self.inc] * self.n
        K = np.zeros((self.n * self.n, self.m * self.n), dtype=FLOAT)
        r = 0
        c = 0
        for k in range(self.n):
            rr, cc = self.inc.shape
            K[r : r + rr, c : c + cc] = self.inc
            r += rr
            c += cc
        B = np.tile(np.eye(self.m, dtype=FLOAT), self.n)
        cvec = np.zeros(self.m * self.n + 1, dtype=FLOAT)
        cvec[-1] = -1.0
        Aeq = np.hstack((K, L.flatten().astype(FLOAT)[:, None]))
        beq = np.zeros(self.n * self.n, dtype=FLOAT)
        Aub = np.hstack((B, np.zeros((self.m, 1), dtype=FLOAT)))
        bub = self.bandwidth
        res = linprog(c=cvec, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs-ds")
        x = res.x.astype(FLOAT)
        flow_vec = x[: self.m * self.n]
        flow = flow_vec.reshape(self.n, self.m).T
        gamma = float(x[-1])
        return Solution(problem=None, flow=flow, gamma=gamma)


