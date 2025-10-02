from typing import List, Tuple

import cvxpy as cp
import networkx as nx
import numpy as np

from src.data.types import FLOAT
from src.solvers.throughput import netoptlibcpp as nol
from src.utils import calculate_laplacian_from_weights_matrix, get_incidence_matrix, get_var_value, get_weights
from src.solvers.edge_based.solution import Solution


def solve_throughput(graph: nx.Graph, traffic_mat: np.ndarray, **solver_kwargs) -> Solution:
    graph = nx.DiGraph(graph)
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")
    incidence_mat = get_incidence_matrix(graph)
    bandwidth = get_weights(graph, "bandwidth")

    flow = cp.Variable((len(graph.edges), traffic_mat.shape[0]))
    gamma = cp.Variable()
    prob = cp.Problem(
        cp.Maximize(gamma),
        [cp.sum(flow, axis=1) <= bandwidth, incidence_mat @ flow == -gamma * traffic_lapl.T, flow >= 0, gamma >= 0],
    )
    prob.solve(**solver_kwargs)

    if prob.status != "optimal":
        gamma = None

    return Solution(problem=prob, flow=get_var_value(flow), gamma=get_var_value(gamma))


def solve_best_throughput_demand(graph: nx.Graph, **solver_kwargs) -> Solution:
    graph = nx.DiGraph(graph)
    incidence_mat = get_incidence_matrix(graph)
    bandwidth = get_weights(graph, "bandwidth")

    n, m = graph.number_of_nodes(), graph.number_of_edges(),
    traffic_mat = cp.Variable((n, n), nonneg=True)
    traffic_lapl = cp.diag(traffic_mat.sum(axis=1)) - traffic_mat 
    flow = cp.Variable((m, n))
    prob = cp.Problem(
        cp.Maximize(traffic_mat.sum()),
        [cp.sum(flow, axis=1) <= bandwidth, incidence_mat @ flow == -traffic_lapl.T, flow >= 0, cp.diag(traffic_mat) == 0],
    )
    prob.solve(**solver_kwargs)

    return Solution(problem=prob, flow=get_var_value(flow), traffic_mat=get_var_value(traffic_mat))


def optimize_throughput(graph: nx.Graph, traffic_mat: np.ndarray, budget: float, **solver_kwargs) -> Solution:
    graph = nx.DiGraph(graph)
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")
    incidence_mat = get_incidence_matrix(graph)
    bandwidth = get_weights(graph, "bandwidth")

    flow = cp.Variable((len(graph.edges), traffic_mat.shape[0]))
    add_bandwidth = cp.Variable(len(graph.edges))
    gamma = cp.Variable()
    prob = cp.Problem(
        cp.Maximize(gamma),
        [
            cp.sum(flow, axis=1) <= bandwidth + add_bandwidth,
            incidence_mat @ flow == -gamma * traffic_lapl.T,
            cp.sum(add_bandwidth) <= budget,
            flow >= 0,
            gamma >= 0,
            add_bandwidth >= 0,
        ],
    )
    prob.solve(**solver_kwargs)

    if prob.status != "optimal":
        gamma = None

    return Solution(
        problem=prob, flow=get_var_value(flow), add_bandwidth=get_var_value(add_bandwidth), gamma=get_var_value(gamma)
    )


def optimize_robust_throughput(
    graph: nx.DiGraph,
    traffic_mat: np.ndarray,
    budget: float,
    proportion_edge_perturbed: float = 1.0,
    **solver_kwargs,
) -> Solution:
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")
    incidence_mat = get_incidence_matrix(graph)
    bandwidth = get_weights(graph, "bandwidth")

    num_edges = len(graph.edges)
    num_nodes = len(graph.nodes)
    flow = [cp.Variable((num_edges, num_nodes)) for _ in range(num_edges)]
    add_bandwidth = cp.Variable(num_edges)
    gamma = cp.Variable(num_edges)
    t = cp.Variable()

    kirchhoff_constr = [(incidence_mat @ f == -gam * traffic_lapl.T) for f, gam in zip(flow, gamma)]
    bandwidth_constr = []
    for e in range(num_edges):
        rhs_mul = np.ones_like(bandwidth)
        rhs_mul[e] = 1 - proportion_edge_perturbed
        bandwidth_constr += [cp.sum(flow[e], axis=1) <= cp.multiply(bandwidth + add_bandwidth, rhs_mul)]

    prob = cp.Problem(
        cp.Maximize(t),
        kirchhoff_constr
        + bandwidth_constr
        + [gamma >= t * np.ones_like(gamma)]
        + [f >= 0 for f in flow]
        + [gamma >= 0, add_bandwidth >= 0, cp.sum(add_bandwidth) <= budget],
    )
    prob.solve(**solver_kwargs)

    if prob.status != "optimal":
        gamma = None

    return Solution(
        problem=prob,
        flow=get_var_value(flow),
        add_bandwidth=get_var_value(add_bandwidth),
        gamma=get_var_value(gamma),
    )


def optimize_robust_throughput_cpp(
    graph: nx.Graph, traffic_mat: np.ndarray, num_deleted_edges: int, budget: float, parallel: bool, logging: bool
):
    graph = nx.DiGraph(graph)
    incidence_mat = nx.incidence_matrix(graph, oriented=True).todense()
    bandwidth = np.array(list(nx.get_edge_attributes(graph, "bandwidth").values()), dtype=FLOAT)
    return nol.optimize_robust_throughput(
        incidence_mat, traffic_mat, bandwidth, num_deleted_edges, budget, parallel, logging
    )
