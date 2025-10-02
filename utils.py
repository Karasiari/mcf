from typing import List, Tuple

import numpy as np


def solve_list_of_problems(solver, budget_list: List[float]) -> Tuple[List[float], List[np.ndarray]]:
    """
    Return result of optimizing robust lambda_2 for sequence of problems
    :solver solver: solver for problems
    :param budget_list: list of float, the list of values for budget for changing of graph
    :return:
       lam2_results: list of float, list of optimized robust lambda 2
       capacities_new_results: list of ndarray of shape (num_edges,), list of optimized vectors of capacities
    """
    lam2_results = []
    capacities_new_results = []
    for budget in budget_list:
        solver.set_budget(budget)
        lam2, capacities_new = solver.solve_problem()
        lam2_results.append(lam2)
        capacities_new_results.append(capacities_new)
    return lam2_results, capacities_new_results
