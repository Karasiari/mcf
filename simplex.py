import networkx as nx
import numpy as np
import scipy.linalg as sla
from scipy.optimize import linprog
from src.solvers.graph_utils.utils import get_edge_attributes
from src.data.load_data import FLOAT
from src.solvers.edge_based.solution import vector_to_solution
from typing import Optional


def get_data_for_lp(graph: nx.DiGraph, traffic_mat: np.ndarray, budget: Optional[float], problem_type: str):
    n = nx.number_of_nodes(graph)
    m = nx.number_of_edges(graph)
    bandwidth = get_edge_attributes(graph, "bandwidth")
    cost = get_edge_attributes(graph, "cost")

    traffic_lapl = np.diag(traffic_mat.sum(axis=1)) - traffic_mat
    incidence_mat = nx.incidence_matrix(graph, oriented=True).todense().astype(FLOAT)
    kirchhoff_constr_mat = sla.block_diag(*([incidence_mat] * n))
    kirchhoff_constr_rhs = -traffic_lapl.flatten()
    bandwidth_constr_mat = np.tile(np.eye(m), n)
    bandwidth_constr_rhs = bandwidth

    if problem_type == "min_cost_concurrent_flow":
        obj_vec = np.tile(cost, n)
        eq_mat = kirchhoff_constr_mat
        eq_rhs = kirchhoff_constr_rhs
        ineq_mat = bandwidth_constr_mat
        ineq_rhs = bandwidth_constr_rhs
    elif problem_type == "opt_network_min_cost_concurrent_flow":
        obj_vec = np.hstack((np.tile(cost, n), np.zeros(m)))
        eq_mat = np.hstack((kirchhoff_constr_mat, np.zeros((n**2, m))))
        eq_rhs = kirchhoff_constr_rhs
        budget_constr_mat = np.hstack((np.zeros(m * n), np.ones(m)))
        ineq_mat = np.vstack((np.hstack((bandwidth_constr_mat, -np.eye(m))), budget_constr_mat))
        ineq_rhs = np.hstack((bandwidth_constr_rhs, np.array([budget])))
    elif problem_type == "max_concurrent_flow":
        obj_vec = np.zeros(m * n + 1)
        obj_vec[-1] = -1.0
        eq_mat = np.hstack((kirchhoff_constr_mat, -kirchhoff_constr_rhs[:, None]))
        eq_rhs = np.zeros(n**2)
        ineq_mat = np.hstack((bandwidth_constr_mat, np.zeros((m, 1))))
        ineq_rhs = bandwidth_constr_rhs
    elif problem_type == "opt_network_max_concurrent_flow":
        obj_vec = np.zeros(m * n + m + 1)
        obj_vec[-1] = -1.0
        eq_mat = np.hstack((kirchhoff_constr_mat, np.zeros((n**2, m)), -kirchhoff_constr_rhs[:, None]))
        eq_rhs = np.zeros(n**2)
        ineq_mat = np.vstack(
            (
                np.hstack((bandwidth_constr_mat, -np.eye(m), np.zeros((m, 1)))),
                np.hstack((np.zeros(m * n), np.ones(m), np.zeros(1))),
            )
        )
        ineq_rhs = np.hstack((bandwidth, np.array([budget])))
    else:
        raise ValueError(f"Unknown problem type '{problem_type}'")

    return (
        obj_vec.astype(FLOAT),
        eq_mat.astype(FLOAT),
        eq_rhs.astype(FLOAT),
        ineq_mat.astype(FLOAT),
        ineq_rhs.astype(FLOAT),
    )


def solve_highs(graph: nx.DiGraph, traffic_mat: np.ndarray, budget: Optional[float], problem_type: str):
    obj_vec, eq_mat, eq_rhs, ineq_mat, ineq_rhs = get_data_for_lp(graph, traffic_mat, budget, problem_type)
    sol_vector = linprog(
        c=obj_vec,
        A_ub=ineq_mat,
        b_ub=ineq_rhs,
        A_eq=eq_mat,
        b_eq=eq_rhs,
        bounds=(0, None),
        method="highs-ds",
    ).x
    return vector_to_solution(graph, sol_vector, problem_type)
