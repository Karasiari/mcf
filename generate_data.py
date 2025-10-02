from typing import List, Optional, Tuple

import fnss
import networkx as nx
import numpy as np


def generate_random_source_target_traffic(
    num_nodes: int,
    distr_type: str = "exponential",
    distr_fun_kvargs: dict = {},
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns source and target volumes for traffic_mat matrix generation
    INPUT:
        num_nodes: amount of nodes in graph
        distr_type: the distribution type string
        distr_fun_kvargs: arguments for distribution function. Does not contain size
    OUTPUT:
        source_volume: a vector of source volumes (ith component value is ith node total output flow)
        target_volume: a vector of target volumes (ith component value is ith node total input flow)
    RAISES:
        NotImplemented error if distr_type not implemended
    """
    supported_types = {
        "exponential": np.random.exponential,
        "lognormal": np.random.lognormal,
    }
    if distr_type in supported_types:
        distr_fun = supported_types[distr_type]
    else:
        raise (
            NotImplementedError(f"distr_type = {distr_type} is not supported yet, use one of {supported_types.keys}")
        )
    source_volume = distr_fun(size=num_nodes, **distr_fun_kvargs)
    target_volume = distr_fun(size=num_nodes, **distr_fun_kvargs)
    # rescale for sum(source_volume) = sum(target_volume)
    source_volume = source_volume * np.sum(target_volume) / np.sum(source_volume)

    return source_volume, target_volume


def generate_random_traffic_mat(
    num_nodes: int,
    distr_type: str = "exponential",
    distr_fun_kvargs: dict = {},
) -> np.ndarray:
    """
    Create a traffic_mat matrix (traffic_mat matrix) for given source and target vectors
    INPUT:
        num_nodes: amount of nodes in graph
        distr_type: a type of distribution for source and target volumes generation
        distr_fun_kvargs: arguments for generating function
                          (depends on implementation see generate_random_source_target_traffic)
    OUTPUT:
        returns a traffic_mat matrix according to:
        X_ij = source_i * target_j / sum(source)
    """
    source_volume, target_volume = generate_random_source_target_traffic(num_nodes, distr_type, distr_fun_kvargs)
    return generate_gravity_model_traffic_mat(np.expand_dims(source_volume, 0), np.expand_dims(target_volume, 0))


def generate_gravity_model_traffic_mat(in_volumes: np.ndarray, out_volumes: np.ndarray) -> np.ndarray:
    """
    INPUT:
        in_volumes: 2D array of incoming flow for every node for every time_stamp
        out_volumes: 2D array of outgoing flow for every node for every time_stamp
    OUTPUT:
        gravity_model_traffic_mats: 3D array of source-target flows for every timestamp built according to gravity model
    """
    total_volume = np.sum(in_volumes, axis=-1)
    assert np.allclose(total_volume, np.sum(out_volumes, axis=-1))
    gravity_model_traffic_mats = np.einsum("ab,ac->abc", out_volumes, in_volumes) / np.expand_dims(
        total_volume, (1, 2)
    )

    return gravity_model_traffic_mats


def generate_ba_wb_random_graph(
    min_nodes_am: int = 3,
    max_nodes_am: int = 100,
    inital_attached_nodes: int = 3,
    edges_for_new_nodes: int = 1,
    custom_bandwidths: Optional[List[int]] = None,
) -> fnss.Topology:
    """
    generate Barabasi-Albert weighted bandwidthed graph
    INPUT:
        min_nodes_am, max_nodes_am: range for random number of nodes
        inital_attached_nodes: number of nodes initially attached to the network
        edges_for_new_nodes: number of edges to attach from a new node to existing nodes
        custom_bandwidths: a list of possible values for bandwidths
    OUTPUT:
        Barabasi-Albert generated graph with weights and bandwidths
    """
    custom_bandwidths = [1] if custom_bandwidths is None else custom_bandwidths
    topology = fnss.barabasi_albert_topology(
        np.random.randint(min_nodes_am, high=max_nodes_am), edges_for_new_nodes, inital_attached_nodes
    )
    fnss.set_capacities_betweenness_gravity(topology, custom_bandwidths)
    fnss.set_weights_inverse_capacity(topology)

    return topology


def generate_fnss_topology_random_traffic_mat(graph: fnss.Topology) -> np.ndarray:
    """
    generate random static traffic mat via fnss for a graph
    INPUT:
        graph: any graph in fnss.Topology with bandwidths
    OUTPUT:
        traffic_mat
    """
    n = len(graph.nodes())
    assert len(graph.capacities()) > 0
    traffic_mat_dict = fnss.static_traffic_matrix(graph, mean=1.0, stddev=0.5, max_u=1.0).flows()
    traffic_mat = np.zeros((n, n))
    for k, v in traffic_mat_dict.items():
        traffic_mat[k] = v
    return traffic_mat


def generate_ba_wb_random_graph_traffic_mat(
    min_nodes_am: int = 3,
    max_nodes_am: int = 100,
    inital_attached_nodes: int = 3,
    edges_for_new_nodes: int = 1,
    custom_bandwidths: Optional[List[int]] = None,
) -> Tuple[nx.DiGraph, np.ndarray]:
    """
    generate Barabasi-Albert weighted bandwidthed graph in networkx with static traffic mat
    INPUT:
        min_nodes_am, max_nodes_am: range for random number of nodes
        inital_attached_nodes: number of nodes initially attached to the network
        edges_for_new_nodes: number of edges to attach from a new node to existing nodes
    OUTPUT:
        Barabasi-Albert generated graph with cost and bandwidth with static traffic mat
    """
    topology = generate_ba_wb_random_graph(
        min_nodes_am, max_nodes_am, inital_attached_nodes, edges_for_new_nodes, custom_bandwidths
    )
    nx_graph = nx.DiGraph(topology)
    # rename capacity into bandwidth, weight into cost
    for _, __, data in nx_graph.edges(data=True):
        data["bandwidth"] = data.pop("capacity")
        data["cost"] = data.pop("weight")
    static_traffic_mat = generate_fnss_topology_random_traffic_mat(topology)
    return nx_graph, static_traffic_mat
