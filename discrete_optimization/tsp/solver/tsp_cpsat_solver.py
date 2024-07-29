#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Optional

from ortools.sat.python.cp_model import CpModel

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import (
    CpSolverSolutionCallback,
    OrtoolsCPSatSolver,
)
from discrete_optimization.tsp.common_tools_tsp import build_matrice_distance
from discrete_optimization.tsp.solver.tsp_solver import SolverTSP
from discrete_optimization.tsp.tsp_model import SolutionTSP, TSPModel

logger = logging.getLogger(__name__)


class CpSatTspSolver(OrtoolsCPSatSolver, SolverTSP):
    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.distance_matrix = build_matrice_distance(
            self.problem.node_count,
            method=self.problem.evaluate_function_indexes,
        )
        self.distance_matrix[self.problem.end_index, self.problem.start_index] = 0

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        current_node = self.problem.start_index
        route_is_finished = False
        path = []
        route_distance = 0
        while not route_is_finished:
            for i in range(self.problem.node_count):
                if i == current_node:
                    continue
                if cpsolvercb.boolean_value(
                    self.variables["arc_literals"][current_node, i]
                ):
                    route_distance += self.distance_matrix[current_node, i]
                    current_node = i
                    if current_node == self.problem.start_index:
                        route_is_finished = True
                    break
            if not route_is_finished:
                path.append(current_node)
        logger.info(f"Recomputed sol length = {route_distance}")
        return SolutionTSP(
            problem=self.problem,
            start_index=self.problem.start_index,
            end_index=self.problem.end_index,
            permutation=path[:-1],
        )

    def init_model(self, **args: Any) -> None:
        model = CpModel()
        num_nodes = self.problem.node_count
        all_nodes = range(num_nodes)
        obj_vars = []
        obj_coeffs = []
        arcs = []
        arc_literals = {}
        for i in all_nodes:
            for j in all_nodes:
                if i == j:
                    continue
                lit = model.new_bool_var(f"{j} follows {i}")
                arcs.append((i, j, lit))
                arc_literals[i, j] = lit
                obj_vars.append(lit)
                obj_coeffs.append(int(self.distance_matrix[i, j]))
        model.add_circuit(arcs)
        if self.problem.start_index != self.problem.end_index:
            model.Add(
                arc_literals[self.problem.end_index, self.problem.start_index] == True
            )
        model.minimize(sum(obj_vars[i] * obj_coeffs[i] for i in range(len(obj_vars))))
        self.variables["arc_literals"] = arc_literals
        self.cp_model = model
