"""Minimal API for a discrete-optimization solver."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations  # see annotations as str

from abc import abstractmethod
from typing import Any, List, Optional, Tuple

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
    Hyperparametrizable,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    fitness_class,
)


class SolverDO(Hyperparametrizable):
    """Base class for a discrete-optimization solver."""

    problem: Problem

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        self.problem = problem
        (
            self.aggreg_from_sol,
            self.aggreg_from_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem,
            params_objective_function=params_objective_function,
        )

    @abstractmethod
    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        """Generic solving function.

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            **kwargs: any argument specific to the solver

        Solvers deriving from SolverDo should use callbacks methods .on_step_end(), ...
        during solve(). But some solvers are not yet updated and are just ignoring it.

        Returns (ResultStorage): a result object containing potentially a pool of solutions
        to a discrete-optimization problem
        """
        ...

    def create_result_storage(
        self, list_solution_fits: Optional[List[Tuple[Solution, fitness_class]]] = None
    ) -> ResultStorage:
        """Create a result storage with the proper mode_optim.

        Args:
            list_solution_fits:

        Returns:

        """
        if list_solution_fits is None:
            list_solution_fits = []
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            mode_optim=self.params_objective_function.sense_function,
        )

    def init_model(self, **kwargs: Any) -> None:
        """Initialize intern model used to solve.

        Can initialize a ortools, milp, gurobi, ... model.

        """
        ...

    def is_optimal(self) -> Optional[bool]:
        """Tell if found solution is supposed to be optimal.

        To be called after a solve.

        Returns:
            optimality of the solution. If information missing, returns None instead.

        """
        return None
