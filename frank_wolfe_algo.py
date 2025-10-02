from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm


def FW_solver(
    linearized_task_solver: Callable[[np.ndarray], Optional[np.ndarray]],
    gradient: Callable[[np.ndarray], np.ndarray],
    init_guess_computer: Callable[[], np.ndarray],
    line_search_solver: Optional[Callable[[np.ndarray, np.ndarray], Optional[float]]] = None,
    step_size: Union[float, str] = "dynamic",
    max_iter: int = 100,
    tolerance: float = 1e-3,
    FW_verbose: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]], Optional[List[float]]]:
    """
    https://fa.bianp.net/blog/2018/notes-on-the-frank-wolfe-algorithm-part-i/

    INPUT:
        linearized_task_solver: solves lin_task with grad at given point with
        gradient: returns a gradient of obj fun at given point
        init_guess_computer: returns inital guess inside a budget space
        line_search_solver: solves line search task at given point in
                            given direction

        fw algo kwargs:
            max_iter: max am of FW algo iters
            step_size: if float -> const step_size if str,
                       'dynamic' stepsize (~ 1/k) or 'line_search'
                        for scipy line_search
            FW_verbose: verbose mode for meta FW algo (not for cvxpy subtasks)
            tolerance: gap absolute value stop line
    OUTPUT:
        Optimal flow matrix, flow values on each iter, grad values on each iter
    """

    iterations = tqdm(range(max_iter)) if FW_verbose else range(max_iter)
    flow_hist = [] if FW_verbose else None
    gap_hist = [] if FW_verbose else None

    if FW_verbose is True:
        print("=================Begin=================")

    # Initial guess should be in budget space
    flow = init_guess_computer()

    if flow is None:
        if FW_verbose is True:
            print("=================InitGuessNoneTerm=================")
        return (None, flow_hist, gap_hist)

    for iter_ind in iterations:
        # Compute gradient
        grad = gradient(flow)
        v = linearized_task_solver(grad)

        if v is None:
            if FW_verbose is True:
                print("=================NoneTerm=================")
                FW_verbose = False
                flow = None
            break

        # Compute gap
        d = v - flow
        g = -1.0 * d @ grad

        # Save currrent flow mat and gap
        if FW_verbose is True:
            flow_hist.append(flow)
            gap_hist.append(g)

        if g < tolerance:
            if FW_verbose is True:
                print("=================TolTerm=================")
                FW_verbose = False
            break

        if FW_verbose is True and iter_ind % 10 == 0:
            # display g in tqdm
            iterations.set_postfix({"current_g": g})
            iterations.update()

        # Choose step size
        if isinstance(step_size, float):
            lr = step_size
        elif isinstance(step_size, str):
            if step_size == "dynamic":
                # lr = min(0.1, (1 / (iter_ind + 1)))
                lr = 2 / (iter_ind + 2)
            elif step_size == "line_search":
                lr = line_search_solver(flow, d)
                if lr is None:
                    # line search failed -> shift from curr point
                    lr = 0.01
            else:
                raise NotImplementedError(f"Incorrect step_size given:{step_size}")
        else:
            raise NotImplementedError(f"Incorrect step_size type given:{type(step_size)}")

        # Perform a step
        flow = flow + lr * d

    if FW_verbose is True:
        print("=================MaxIterTerm=================")

    return flow, flow_hist, gap_hist
