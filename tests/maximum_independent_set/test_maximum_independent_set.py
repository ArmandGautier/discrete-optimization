import logging

import pytest

from discrete_optimization.maximum_independent_set.mis_model import MisProblem
from discrete_optimization.maximum_independent_set.mis_parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.mis_solvers import solve, solvers_map
from discrete_optimization.maximum_independent_set.solvers.mis_gurobi import (
    MisMilpSolver,
    MisQuadraticSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_kamis import (
    MisKamisSolver,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True

kamis_available = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("solver_class", solvers_map)
def test_solvers(solver_class):
    if (
        solver_class == MisMilpSolver or solver_class == MisQuadraticSolver
    ) and not gurobi_available:
        pytest.skip("You need Gurobi to test this solver.")
    if solver_class == MisKamisSolver and not kamis_available:
        pytest.skip("You need Kamis to test this solver.")
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    results = solve(
        method=solver_class, problem=mis_model, **solvers_map[solver_class][1]
    )
    sol, fit = results.get_best_solution_fit()
    assert mis_model.satisfy(sol)
