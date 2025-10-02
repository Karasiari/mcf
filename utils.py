from typing import Optional, Union

import cvxpy as cp
import networkx as nx
import numpy as np
import scipy
from scipy.sparse import csr_matrix

from src.data.types import FLOAT


def get_weights_matrix(graph: nx.DiGraph, key: str = "bandwidth") -> np.ndarray:
    """
    Extract capacity matrix from graph
    :param graph: nx.DiGraph,: graph with attribute cost on edges
    :param key: str, default="bandwidth", name of attribute to obtain weights matrix
    :return:
        capacity_matrix: ndarray of shape (num_nodes, num_nodes)
    """
    capacity_matrix = nx.adjacency_matrix(graph, weight=key)
    capacity_matrix = capacity_matrix.toarray()
    return capacity_matrix


def get_incidence_matrix(graph: nx.DiGraph) -> np.ndarray:
    """
    Construct incidence matrix
    :param graph: nx.DiGraph,: graph with attribute cost on edges
    :return:
        incidence_matrix: ndarray of shape (num_nodes, num_edges), incidence matrix
    """
    incidence_matrix = nx.incidence_matrix(graph, edgelist=graph.edges, oriented=True)
    incidence_matrix = incidence_matrix.toarray()
    return incidence_matrix


def get_weights_using_weights_matrix(graph: nx.DiGraph, key: str = "bandwidth") -> np.ndarray:
    """
    Construct extract all weights
    :param graph: nx.DiGraph,: graph with attribute cost on edges
    :param key: str, default="bandwidth", name of attribute to obtain weights matrix
    :return:
        capacities: ndarray of shape (num_nodes), all capacities in graph
    """
    weights_matrix = get_weights_matrix(graph, key=key)
    capacities = np.triu(weights_matrix)
    capacities = capacities[capacities != 0]
    return capacities


def get_weights(graph: nx.DiGraph, key: str) -> np.ndarray:
    """
    Extract edge weights
    :param graph: nx.DiGraph, graph with weights on edges
    :param key: str, name of attribute to obtain weights
    :return:
        weights: ndarray of shape (num_nodes), all edge weights in graph
    """
    return np.array(list(nx.get_edge_attributes(graph, key).values()), dtype=FLOAT)


def calculate_laplacian(graph: nx.DiGraph, lapl_type: str, key: str = "bandwidth") -> np.ndarray:
    """
    Calculate laplacian matrix: L := D - W, where D is diagonal matrix with d_{ii} = sum_j w_ij
    if lapl_type == 'in' else d_{ii} = sum_j w_ji
    :param graph: nx.DiGraph,: graph with attribute cost on edges
    :param key: str, default="bandwidth", name of attribute to obtain weights matrix
    :return:
        L: ndarray of shape (num_nodes, num_nodes), laplacian matrix for input weight matrix
    """
    weights_matrix = get_weights_matrix(graph, key=key)
    return calculate_laplacian_from_weights_matrix(weights_matrix, typ=lapl_type)


def calculate_laplacian_from_weights_matrix(weights_matrix: np.ndarray, typ: str) -> np.ndarray:
    """
    Calculate laplacian matrix: L := D - W, where D is diagonal matrix with out-degrees or in degrees
    depending on typ. If typ == 'out', then d_ii = sum_j w_ij; if typ == 'in', then d_ii = sum_j w_ji.
    :param weights_matrix: ndarray of shape (num_nodes, num_nodes), symmetric weights matrix of graph
    :param typ: str, 'out' to compute out-degree laplacian, 'in' to compute in-degree laplacian
    :return:
        L: ndarray of shape (num_nodes, num_nodes), laplacian matrix for input weight matrix
    """
    axis = 0 if typ == "in" else 1
    return np.diag(weights_matrix.sum(axis)) - weights_matrix


def get_var_value(var: Optional[cp.Variable]) -> Optional[float | np.ndarray]:
    """
    Get cvxpy.Variable value if not None
    :param var: cvxpy.Variable, variable to extract value
    :return:
        value: var.value if var is not None, else None
    """
    return var.value if var is not None else None
