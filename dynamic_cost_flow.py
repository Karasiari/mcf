from typing import Callable, List, Optional, Tuple, Union

import cvxpy as cp
import networkx as nx
import numpy as np

from src.data.types import FLOAT
from src.solvers.edge_based.frank_wolfe_algo import FW_solver
from src.solvers.graph_utils.shortest_paths_gt import (
    flows_on_shortest_gt,
    get_graphtool_graph,
    maybe_create_and_get_ep,
)


def gradient(flow: np.ndarray, cost: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
    """
    gradient of dynamic costs task objective
    """

    grad_e = cost / ((1.0 - flow / bandwidth) ** 2)

    # Can't use inf here, because graph-tool interprets inf as no edge and fails to find a path
    grad_e[flow / bandwidth >= 1] = 1e6

    return grad_e


def objective(flow: np.ndarray, cost: np.ndarray, bandwidth: np.ndarray) -> float:
    """
    dynamic costs tasks objective
    """

    return (
        np.inf
        if np.any(flow >= bandwidth)
        else cost @ (np.multiply(bandwidth, np.reciprocal((1 - flow / bandwidth)) - 1))
    )


def solve_dynamic_costs_FW(
    graph_nx: nx.Graph,
    traffic_mat: np.ndarray,
    cost_alpha: float = 1.0,
    step_size: Union[float, str] = "dynamic",
    max_iter: int = 100,
    tolerance: float = 1e-3,
    FW_verbose: bool = False,
    plot_g_fun: Optional[Callable[[List], None]] = None,
    **cp_solver_kwargs,
) -> Tuple[np.ndarray, Optional[List[np.ndarray]], Optional[List[float]]]:
    """
    https://fa.bianp.net/blog/2018/notes-on-the-frank-wolfe-algorithm-part-i/
    Frank-Wolfe Optimization Algorithm for a task with dynamic costs function:
    cost(x)_i = x_i * staticCost_i / (1 - x_i / bandwidth_i)

    bandwidth: (m, ) vector
    cost: (m, ) vector
    traffic_lapl: (n, n) matrix
    incidence_mat: (n, m) matrix

    INPUT:
        graph: directed graph of transport problem
        traffic_mat: correspondence matrix
        cost_alpha: mul costs by a value
        plot_g_fun: a function to plot gap history
        fw_kwargs_dict: arguments for FW algo (see FW_solver docstring)
        cp_solver_kwargs: named args for cvxopt subtasks
    OUTPUT:
        Optimal flow matrix, flow values on each iter, grad values on each iter
    """

    m, n = len(graph_nx.edges), traffic_mat.shape[0]
    graph_nx = nx.DiGraph(graph_nx)
    graph_gt = get_graphtool_graph(graph_nx)
    traffic_lapl = np.diag(traffic_mat.sum(axis=1)) - traffic_mat
    incidence_mat = nx.incidence_matrix(graph_nx, oriented=True).todense()

    bandwidth = np.array(list(nx.get_edge_attributes(graph_nx, "bandwidth").values()), dtype=FLOAT)
    cost = np.array(list(nx.get_edge_attributes(graph_nx, "cost").values()), dtype=FLOAT)
    cost = cost * cost_alpha

    def linearized_task_solver_cvxpy(grad: np.ndarray) -> Optional[np.ndarray]:
        v = cp.Variable((m, n))
        prob = cp.Problem(
            cp.Minimize(cp.trace(v @ grad.T)),
            [
                cp.sum(v, axis=1) <= bandwidth,
                (incidence_mat @ v).T == -traffic_lapl,
                v >= 0,
            ],
        )
        prob.solve(**cp_solver_kwargs)
        if prob.status != "optimal":
            v = None
        return v.value if v is not None else None

    def linearized_task_solver_dijkstra(grad: np.ndarray) -> np.ndarray:
        weights = maybe_create_and_get_ep(graph_gt, grad)
        return flows_on_shortest_gt(graph_gt, traffic_mat=traffic_mat, weights=weights)

    def init_guess_computer() -> Optional[np.ndarray]:
        """
        solving a task with const objective
        """
        return linearized_task_solver_dijkstra(np.zeros(m))

    def line_search_solver(x: np.ndarray, direction: np.ndarray) -> Optional[float]:
        def golden_search(f, left, right, eps):
            """If f(l_gold) == f(r_gold), moves towards left"""

            tau = (1 + np.sqrt(5)) / 2

            def left_gold(l, r):
                return l + (1 / tau**2) * (r - l)

            def right_gold(l, r):
                return l + (1 / tau) * (r - l)

            l_gold = left_gold(left, right)
            r_gold = right_gold(left, right)
            l_gold_val = f(l_gold)
            r_gold_val = f(r_gold)

            while right - left > eps:
                if l_gold_val <= r_gold_val:
                    right = r_gold
                    r_gold = l_gold
                    r_gold_val = l_gold_val
                    l_gold = left_gold(left, right)
                    l_gold_val = f(l_gold)
                    continue

                left = l_gold
                l_gold = r_gold
                l_gold_val = r_gold_val
                r_gold = right_gold(left, right)
                r_gold_val = f(r_gold)

            return l_gold

        def objective_1D(step):
            return objective(x + step * direction, cost=cost, bandwidth=bandwidth)

        assert objective_1D(0) < np.inf, "Previously found point (probably, init guess) is infeasible"

        step = golden_search(objective_1D, 0, 1, eps=1e-5)
        assert (
            objective_1D(step) < np.inf
        ), f"Linesearch found infeasible point: {step=:.3f}, {objective_1D(step)=:.3f}, {objective_1D(0)=:.3f}, {objective_1D(0.001)=:.3f}, {objective_1D(1)=:.3f}"

        return step

    flow, flow_hist, gap_hist = FW_solver(
        linearized_task_solver=linearized_task_solver_dijkstra,
        gradient=lambda x: gradient(x, cost=cost, bandwidth=bandwidth),
        init_guess_computer=init_guess_computer,
        line_search_solver=line_search_solver,
        step_size=step_size,
        max_iter=max_iter,
        tolerance=tolerance,
        FW_verbose=FW_verbose or plot_g_fun is not None,
    )

    if plot_g_fun is not None:
        plot_g_fun(gap_hist)

    return flow, flow_hist, gap_hist


def solve_dynamic_costs_cvxpy(
    graph: nx.Graph, traffic_mat: np.ndarray, cost_alpha: float = 1.0, **solver_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traffic_lapl = np.diag(traffic_mat.sum(axis=1)) - traffic_mat
    incidence_mat = nx.incidence_matrix(graph, oriented=True).todense()

    bandwidth = np.array(list(nx.get_edge_attributes(graph, "bandwidth").values()), dtype=FLOAT)
    cost = np.array(list(nx.get_edge_attributes(graph, "cost").values()), dtype=FLOAT)
    cost = cost * cost_alpha

    flow = cp.Variable((len(graph.edges), traffic_mat.shape[0]))

    flow_on_edge = cp.sum(flow, axis=1)
    obj = cost @ (cp.multiply(bandwidth, cp.inv_pos(1 - flow_on_edge / bandwidth) - 1))
    prob = cp.Problem(
        cp.Minimize(obj),
        [
            flow_on_edge <= bandwidth,
            (incidence_mat @ flow).T == -traffic_lapl,
            flow >= 0,
        ],
    )
    prob.solve(**solver_kwargs)
    if prob.status != "optimal":
        flow = None

    flow = flow.value if flow is not None else None
    costs, potentials, nonneg_duals = [cons.dual_value for cons in prob.constraints]
    return flow, costs, potentials, nonneg_duals
