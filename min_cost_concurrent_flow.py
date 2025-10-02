from typing import Dict, Tuple

import cvxpy as cp
import networkx as nx
import numpy as np

from src.data.types import FLOAT
from src.utils import calculate_laplacian_from_weights_matrix, get_incidence_matrix, get_var_value, get_weights
from src.solvers.edge_based.solution import Solution


def solve_min_cost_concurrent_flow(
    graph: nx.Graph, traffic_mat: np.ndarray, **solver_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    graph = nx.DiGraph(graph)
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")
    incidence_mat = get_incidence_matrix(graph)

    bandwidth = get_weights(graph, "bandwidth")
    cost = get_weights(graph, "cost")

    flow = cp.Variable((len(graph.edges), traffic_mat.shape[0]))
    prob = cp.Problem(
        cp.Minimize(cp.sum(flow, axis=1) @ cost),
        [cp.sum(flow, axis=1) <= bandwidth, incidence_mat @ flow == -traffic_lapl.T, flow >= 0],
    )
    prob.solve(**solver_kwargs)
    return Solution(problem=prob, flow=get_var_value(flow))


def opt_network_min_cost_concurrent_flow(
    graph: nx.DiGraph, traffic_mat: np.ndarray, budget: float, **solver_kwargs
) -> Tuple:
    graph = nx.DiGraph(graph)
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")
    incidence_mat = get_incidence_matrix(graph)

    bandwidth = get_weights(graph, "bandwidth")
    cost = get_weights(graph, "cost")

    flow = cp.Variable((len(graph.edges), traffic_mat.shape[0]))
    add_bandwidth = cp.Variable(len(graph.edges))
    prob = cp.Problem(
        cp.Minimize(cp.sum(flow, axis=1) @ cost),
        [
            cp.sum(flow, axis=1) <= bandwidth + add_bandwidth,
            incidence_mat @ flow == -traffic_lapl.T,
            cp.sum(add_bandwidth) <= budget,
            flow >= 0,
            add_bandwidth >= 0,
        ],
    )
    prob.solve(**solver_kwargs)
    return Solution(problem=prob, flow=get_var_value(flow), add_bandwidth=get_var_value(add_bandwidth))


def get_flow_cost_and_amount(flow: np.ndarray, graph: nx.Graph) -> Dict:
    """
    Given a feasible point (flow on every edge + graph topology with capacities)
    returns the cost of the point according to the formula:
        sum_{e in E} f_e * c_e,
        where e is edge and E is a set of all edges c_e, f_e is a cost & flow on e
    INPUT:
        flow: amount of flow for every edge that does
              not conflict with bandwidth restrictions of graph
        graph: a topology with costs and bandwidth restrictions
               on every edge
    OUTPUT:
        Dictionary:
            flow_cost: the total cost of given flow
            flow_amount: sum of all demands in
                         the transport task
    """
    graph = nx.DiGraph(graph)
    incidence_mat = get_incidence_matrix(graph)
    edge_cost = np.array(list(nx.get_edge_attributes(graph, "cost").values()), dtype=FLOAT)
    flow_cost = np.sum(flow, axis=1) @ edge_cost
    flow_amount = np.abs(flow * incidence_mat.T).sum()
    return dict(flow_cost=flow_cost, flow_amount=flow_amount)
