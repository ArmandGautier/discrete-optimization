from typing import List

import numpy as np

from discrete_optimization.tsp.tsp_model import Point2D, length


def length_1(point1, point2):
    return abs(point1.x - point2.x) + abs(point1.y - point2.y)


def compute_length(solution, list_points, nodeCount):
    obj = length(list_points[solution[-1]], list_points[solution[0]])
    lengths = []
    pp = obj
    for index in range(0, nodeCount - 1):
        ll = length(list_points[solution[index]], list_points[solution[index + 1]])
        obj += ll
        lengths += [ll]
    lengths += [pp]
    return lengths, obj


def baseline_in_order(nodeCount: int, points: List[Point2D]):
    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])
    return solution, obj, 0


def build_matrice_distance(nodeCount: int, points: List[Point2D], method=None):
    if method is None:
        method = length
    matrix = np.zeros((nodeCount, nodeCount))
    for i in range(nodeCount):
        for j in range(i + 1, nodeCount):
            d = method(i, j)
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix


def build_matrice_distance_np(nodeCount: int, points: List[Point2D]):
    matrix_x = np.ones((nodeCount, nodeCount), dtype=np.int32)
    matrix_y = np.ones((nodeCount, nodeCount), dtype=np.int32)
    print("matrix init")
    for i in range(nodeCount):
        matrix_x[i, :] *= int(points[i].x)
        matrix_y[i, :] *= int(points[i].y)
    print("multiplied done")
    matrix_x = matrix_x - np.transpose(matrix_x)
    matrix_y = matrix_y - np.transpose(matrix_y)
    distances = np.abs(matrix_x) + np.abs(matrix_y)
    sorted_distance = np.argsort(distances, axis=1)
    print(sorted_distance.shape)
    return sorted_distance, distances


def closest_greedy(nodeCount: int, points: List[Point2D]):
    sd, d = build_matrice_distance_np(nodeCount, points)
    sol = [0]
    length_circuit = 0.0
    index_in_sol = {0}
    s = sd[:, 1:]
    cur_point = 0
    nb_point = 1
    while nb_point < nodeCount:
        n = next(p for p in s[cur_point, :] if p not in index_in_sol)
        length_circuit += length(points[cur_point], points[n])
        index_in_sol.add(n)
        sol += [n]
        cur_point = n
        nb_point += 1
    length_circuit += length(points[cur_point], points[0])
    return sol, length_circuit, 0
