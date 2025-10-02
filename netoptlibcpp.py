# Networks optimization library

import ctypes
import os
import sys

import numpy as np


def py_list2d_to_c_array2d(py_list, rows, cols):
    row_type = ctypes.c_double * cols
    result_type = ctypes.POINTER(ctypes.c_double) * rows
    result = result_type()
    for i in range(rows):
        row = row_type()
        for j in range(cols):
            row[j] = py_list[i][j]
        result[i] = ctypes.cast(row, ctypes.POINTER(ctypes.c_double))
    return ctypes.cast(result, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))


def py_list1d_to_c_array1d(py_list, size_):
    v_type = ctypes.c_double * size_
    result = v_type()
    for i in range(size_):
        result[i] = py_list[i]
    return ctypes.cast(result, ctypes.POINTER(ctypes.c_double))


class OptimizationResult(ctypes.Structure):
    _fields_ = [
        ("feasible", ctypes.c_bool),
        ("value", ctypes.c_double),
        ("solutionVector", ctypes.POINTER(ctypes.c_double)),
        ("svSize", ctypes.c_int),
        ("bestIndices", ctypes.POINTER(ctypes.c_int)),
        ("db", ctypes.POINTER(ctypes.c_double)),
        ("dbSize", ctypes.c_int),
    ]


def load_dynamic_library():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if sys.platform == "win32":
        dll_path = os.path.join(current_dir, "networks_optimization_lib.dll")
        cpplib = ctypes.WinDLL(dll_path, winmode=0)
    elif sys.platform == "darwin":
        cpplib = ctypes.CDLL(os.path.join(current_dir, "networks_optimization_lib.dylib"))
    else:
        cpplib = ctypes.CDLL(os.path.join(current_dir, "networks_optimization_lib.so"))
    return cpplib


def optimize_robust_throughput_from_json(json_path, schema_pas, q, budget, parallel, logging):
    cpplib = load_dynamic_library()

    cpplib.optimizeRobustThroughputFromJson.restype = ctypes.POINTER(OptimizationResult)
    cpplib.optimizeRobustThroughputFromJson.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_bool,
        ctypes.c_bool,
    ]
    encoded_json_path = json_path.encode("utf-8")
    encoded_schema_path = schema_pas.encode("utf-8")
    result_ptr = cpplib.optimizeRobustThroughputFromJson(
        encoded_json_path, encoded_schema_path, q, budget, parallel, logging
    )

    solution = result_ptr.contents
    best_indices = [solution.bestIndices[i] for i in range(q)]
    flow_vector = [solution.solutionVector[i] for i in range(solution.svSize)]
    db = [solution.db[i] for i in range(solution.dbSize)]
    if solution.feasible:
        print("feasible")
    else:
        print("infeasible")
    print("Objective value:", solution.value)
    print("Best indices:", best_indices)
    print("Flow:")
    formatted_arr = ", ".join(["{:.2f}".format(num) for num in flow_vector])
    print("[" + formatted_arr + "]")
    print("db:")
    formatted_arr = ", ".join(["{:.2f}".format(num) for num in db])
    print("[" + formatted_arr + "]")

    cpplib.freeMemory.argtypes = [ctypes.POINTER(OptimizationResult)]
    cpplib.freeMemory(result_ptr)


def optimize_robust_throughput(incidence, demands, bandwidth, q, budget, parallel, logging):
    cpplib = load_dynamic_library()

    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")

    cpplib.optimizeRobustThroughput.restype = ctypes.POINTER(OptimizationResult)
    cpplib.optimizeRobustThroughput.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ND_POINTER_1,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_bool,
        ctypes.c_bool,
    ]

    n, m = incidence.shape

    c_incidence = py_list2d_to_c_array2d(incidence.tolist(), n, m)
    c_demands = py_list2d_to_c_array2d(demands.tolist(), n, n)

    result_ptr = cpplib.optimizeRobustThroughput(c_incidence, c_demands, bandwidth, n, m, q, budget, parallel, logging)

    solution = result_ptr.contents
    feasible = solution.feasible
    value = solution.value
    best_indices = [solution.bestIndices[i] for i in range(q)]
    flow_vector = [solution.solutionVector[i] for i in range(solution.svSize)]
    db = [solution.db[i] for i in range(solution.dbSize)]

    cpplib.freeMemory.argtypes = [ctypes.POINTER(OptimizationResult)]
    cpplib.freeMemory(result_ptr)

    return [feasible, value, flow_vector, best_indices, db]
