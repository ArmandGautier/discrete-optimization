import os
import logging

import pandas
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel

os.environ["DO_SKIP_MZN_CHECK"] = "1"

import numpy as np
from examples.qiskit_examples.personnalized_ansatz import quantum_personnalized_QAOA, quantum_personnalized_VQE
from discrete_optimization.facility.solvers.facility_quantum import VQEFacilitySolver
from examples.qiskit_examples.facility_example import facility_example
from discrete_optimization.facility.solvers.facility_lp_solver import LP_Facility_Solver
from examples.qiskit_examples.knapsack_example import knapsack_example
from examples.qiskit_examples.general_QAOA_VQE_example import quantum_generalQAOA, quantum_generalVQE
from discrete_optimization.knapsack.solvers.knapsack_quantum import VQEKnapsackSolver, QAOAKnapsackSolver
#from examples.qiskit_examples.facility_example import facility_example
from examples.qiskit_examples.TSP_example import tsp_example
from qiskit_algorithms.utils import validate_initial_point, validate_bounds
from discrete_optimization.tsp.solver.solver_ortools import TSP_ORtools
from qiskit_optimization import QuadraticProgram
from discrete_optimization.tsp.tsp_model import Point2D, TSPModel2D
from discrete_optimization.tsp.solver.tsp_quantum import QAOATSPSolver, VQETSPSolver

from discrete_optimization.generic_tools.qiskit_tools import get_result_from_dict_result, GeneralVQESolver
from qiskit_aer import AerSimulator
from qiskit_optimization.converters import QuadraticProgramToQubo
from scipy.optimize import minimize
from qiskit_ibm_runtime import EstimatorV2 as Estimator

from discrete_optimization.coloring.solvers.coloring_quantum import QAOAColoringSolver_MinimizeNbColor, \
    QAOAColoringSolver_FeasibleNbColor, VQEColoringSolver_MinimizeNbColor, VQEColoringSolver_FeasibleNbColor

from qiskit import QuantumCircuit
from examples.qiskit_examples.coloring_example import quantum_coloring
from examples.qiskit_examples.mis_example import quantum_mis
from qiskit.circuit import Gate
from qiskit.circuit.library import IQP, HamiltonianGate, QAOAAnsatz
from qiskit.primitives import Sampler
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms import QAOA
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from discrete_optimization.knapsack.knapsack_model import Item, KnapsackModel

from qiskit_algorithms.optimizers import SPSA, COBYLA, BOBYQA
from discrete_optimization.facility.facility_model import Facility, Point, Customer, FacilityProblem, \
    FacilityProblem2DPoints, FacilitySolution

import networkx as nx
from discrete_optimization.maximum_independent_set.mis_plot import plot_mis_solution, plot_mis_graph
from tests.maximum_independent_set.test_maximum_independent_set import test_solvers
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.coloring.coloring_model import ColoringProblem
from discrete_optimization.maximum_independent_set.solvers.mis_ortools import MisOrtoolsSolver
from discrete_optimization.maximum_independent_set.solvers.mis_quantum import QAOAMisSolver, VQEMisSolver
from discrete_optimization.datasets import fetch_data_for_mis
from discrete_optimization.maximum_independent_set.solvers.mis_gurobi import MisMilpSolver
from discrete_optimization.maximum_independent_set.mis_model import MisSolution, MisProblem
from discrete_optimization.maximum_independent_set.mis_parser import dimacs_parser, dimacs_parser_nx, get_data_available

for name in logging.root.manager.loggerDict:
    if name.startswith("qiskit"):
        logging.getLogger(name).setLevel(logging.WARNING)

if __name__ == "__main__":
    #fetch_data_for_mis()
    """
    file = [f for f in get_data_available() if "1dc.256" in f][0]
    misProblem = dimacs_parser_nx(file)
    misSolver = MisMilpSolver(problem=misProblem)
    res = misSolver.solve(time_limit=20)
    sol, fit = res.get_best_solution_fit()
    sol: MisSolution
    print(fit)
    
    plot_mis_solution(sol)

    nodes_chosen = []
    for i in range(0, len(sol.chosen)):
        if sol.chosen[i] == 1:
            nodes_chosen.append(misProblem.index_to_nodes[i])
    print(nodes_chosen)
    print(len(nodes_chosen))

    """

    """
    graph = nx.Graph()

    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)
    graph.add_edge(2, 6)
    graph.add_edge(3, 4)
    graph.add_edge(3, 5)
    graph.add_edge(4, 5)
    graph.add_edge(4, 6)

    misProblem = MisProblem(graph)
    plot_mis_graph(misProblem)
    misSolver = MisMilpSolver(problem=misProblem)
    misSolver.init_model()
    res = misSolver.solve()

    sol, fit = res.get_best_solution_fit()
    sol: MisSolution

    plot_mis_solution(sol)
    """


    """
    nodes = [(1, {}), (2, {}), (3, {}), (4, {})]
    edges = [(1, 2, {}), (1, 3, {}), (2, 4, {}), (3, 4, {})]

    #nodes = [(1, {}), (2, {}), (3, {})]
    #edges = [(1, 2, {}), (1, 3, {})]

    coloringProblem = ColoringProblem(Graph(nodes=nodes, edges=edges))
    coloringSolver = VQEColoringSolver_FeasibleNbColor(coloringProblem, nb_color=2)
    coloringSolver.init_model()
    kwargs = {}
    res = coloringSolver.solve(**kwargs)
    sol, _ = res.get_best_solution_fit()
    print(sol)
    print(coloringProblem.satisfy(sol))

    coloringSolver = QAOAColoringSolver_FeasibleNbColor(coloringProblem, nb_color=2)
    coloringSolver.init_model()
    kwargs = {"reps": 4}
    res = coloringSolver.solve(**kwargs)
    sol, _ = res.get_best_solution_fit()
    print(sol)
    print(coloringProblem.satisfy(sol))
    """
    """

    max_capacity = 10

    i1 = Item(0, 4, 2)
    i2 = Item(1, 5, 2)
    i3 = Item(2, 4, 3)
    i4 = Item(3, 2, 1)
    i5 = Item(4, 5, 3)
    i6 = Item(5, 2, 1)

    knapsackProblem = KnapsackModel([i1, i2, i3, i4, i5, i6], max_capacity)
    knapsackSolver = QAOAKnapsackSolver(knapsackProblem)
    kwargs = {}
    knapsackSolver.init_model()
    res = knapsackSolver.solve(**kwargs)
    print(knapsackSolver.ansatz.draw())
    sol, _ = res.get_best_solution_fit()
    print(sol)
    """


    #quantum_mis()
    #knapsack_example()



    """
    f1 = Facility(0, 2, 5, Point(1, 1))
    f2 = Facility(1, 1, 2, Point(-1, -1))

    c1 = Customer(0, 2, Point(2, 2))
    c2 = Customer(1, 5, Point(0, -1))

    facilityProblem = FacilityProblem2DPoints(2, 2, [f1, f2], [c1, c2])
    facilitySolver = QAOAFacilitySolver(facilityProblem)
    facilitySolver.init_model()
    optimizer = SPSA(maxiter=250)
    res = facilitySolver.solve(optimizer=optimizer)
    sol, _ = res.get_best_solution_fit()
    print(sol)
    """
    """
    service = QiskitRuntimeService(channel="ibm_quantum",
                                   token="745343ef46b4403efeb271640b3c07998f160881537eb953266f20a08bf7abd012ea3a572c301dce7f8964645da1733f723e1972739c98320236fa9047def6c8")
    backend = service.least_busy(operational=True, simulator=False)

    qubo = QuadraticProgram()
    qubo.binary_var("x1")
    qubo.binary_var("x2")
    qubo.minimize(linear={"x1": 5, "x2": -1}, quadratic={("x1", "x2"): -2})

    op, offset = qubo.to_ising()
    circuit = QuantumCircuit(op.num_qubits)
    circuit.h(0)
    circuit.h(1)
    circuit.append(HamiltonianGate(op, 1), [0, 1])
    circuit.measure_all()

    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)
    ansatz = pm.run(circuit)

    session = Session(backend=backend)

    sampler = SamplerV2(session=session)

    job = sampler.run([ansatz])
    result_sim = job.result()
    print(result_sim[0].data.meas.get_counts())
    """

    """
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance='ibm-q/open/main',
        token='745343ef46b4403efeb271640b3c07998f160881537eb953266f20a08bf7abd012ea3a572c301dce7f8964645da1733f723e1972739c98320236fa9047def6c8'
    )
    job = service.job('csbgdyfd8m00008zb6x0')
    job_result = job.result()
    dict_res = job_result[0].data.meas.get_counts()
    print(dict_res)
    m = 0
    k = None
    for key, value in dict_res.items():
        if value > m:
            m = value
            k = key
    res = list(k)
    print(res)
    res.reverse()
    print(res)
    res = [int(x) for x in res]
    print(res)
    """




    #test_solvers(MisOrtoolsSolver)
    """




    graph = nx.Graph()

    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)
    graph.add_edge(2, 6)
    graph.add_edge(3, 4)
    graph.add_edge(3, 5)
    graph.add_edge(4, 5)
    graph.add_edge(4, 6)

    # we create an instance of MisProblem
    misProblem = MisProblem(graph)
    # we create an instance of a QAOAMisSolver
    misSolver = QAOAMisSolver(misProblem)
    # we initialize the solver, in fact this step transform the problem in a QUBO formulation
    misSolver.init_model()

    #service = QiskitRuntimeService(channel="ibm_quantum")
    #service = QiskitRuntimeService(channel="ibm_quantum", token="223d2a7c1428c764cb33aeec33b2cdce52958ff7c28d49299cfb0e12d69f902daceef6495050a972223f167aa830876d99585bcfb205b78c170b3535ceffd84e")
    #backend_temp = service.backend("ibm_kyoto")
    #noise_model = NoiseModel.from_backend(backend_temp)
    backend = AerSimulator()
    #print(getattr(backend.options, "noise_model"))
    #print(backend)
    kwargs = {"backend": backend, "maxiter": 300, "reps": 2, "nb_shots": 10000}
    res = misSolver.solve(**kwargs)
    sol, _ = res.get_best_solution_fit()
    print(misSolver.ansatz.draw())
    print(misSolver.executed_ansatz.draw())
    print(misSolver.final_ansatz.draw())
    circ = misSolver.final_ansatz
    state = Statevector.from_int(0, 2 ** 6)
    state = state.evolve(circ)
    print(str(state.draw()))
    print(sol)
    plot_mis_solution(sol)
    
    """


    

    

    # TODO tester knapsack avec item avec grosse value
    # TODO optimiser nb slack variable for knapsack

    #quantum_generalVQE()
    #quantum_coloring()
    #facility_example()
    #quantum_generalQAOA()


    """

    quadratic_program = QuadraticProgram()

    var_names = {}
    for i in range(0, 5):
        x_new = quadratic_program.binary_var("x" + str(i))
        var_names[i] = x_new.name

    constant = 0
    linear = {}
    quadratic = {}

    linear[var_names[0]] = 6
    linear[var_names[1]] = 4
    linear[var_names[2]] = 8
    linear[var_names[3]] = 5
    linear[var_names[4]] = 5

    quadratic_program.maximize(constant, linear, quadratic)

    c1 = {var_names[0]: 2, var_names[1]: 2, var_names[2]: 4, var_names[3]: 3, var_names[4]: 2}
    quadratic_program.linear_constraint(c1, "<=", 7)

    c2 = {var_names[0]: 1, var_names[1]: 2, var_names[2]: 2, var_names[3]: 1, var_names[4]: 2}
    quadratic_program.linear_constraint(c2, "=", 4)

    c3 = {var_names[0]: 3, var_names[1]: 3, var_names[2]: 2, var_names[3]: 4, var_names[4]: 4}
    quadratic_program.linear_constraint(c3, ">=", 5)

    conv = QuadraticProgramToQubo()
    qubo = conv.convert(quadratic_program)
    print(qubo.get_num_vars())
    print(qubo.objective)
    print(qubo.variables)
    
    hamiltonian, offset = qubo.to_ising()
    ansatz = QAOAAnsatz(hamiltonian, reps=2)

    backend = AerSimulator()
    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)
    ansatz = pm.run(ansatz)
    hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    session = Session(backend=backend)

    estimator = Estimator(backend=backend, session=session)

    sampler = SamplerV2(session=session)
    
    """


    def cost_func(params, ansatz, hamiltonian, estimator):
        """Return estimate of energy from estimator

        Parameters:
            params (ndarray): Array of ansatz parameters
            ansatz (QuantumCircuit): Parameterized ansatz circuit
            hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
            estimator (EstimatorV2): Estimator primitive instance
            callback_dict: dictionary for storing intermediate results

        Returns:
            float: Energy estimate
        """
        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()
        cost = result[0].data.evs[0]

        return cost
    """
    res = minimize(
        cost_func,
        validate_initial_point(point=None, circuit=ansatz),
        args=(ansatz, hamiltonian, estimator),
        method="COBYLA",
        bounds=validate_bounds(ansatz),
    )

    # Assign solution parameters to our ansatz
    qc = ansatz.assign_parameters(res.x)
    # Add measurements to our circuit
    qc.measure_all()
    # transpile our circuit
    qc = pm.run(qc)
    # run our circuit with optimal parameters find at the minimization step
    results = sampler.run([qc]).result()
    best_result = get_result_from_dict_result(results[0].data.meas.get_counts())
    print(best_result)
    """

    def matrix(quad: QuadraticProgram):
        num_var = quad.get_num_vars()
        m = np.zeros((num_var, num_var))
        obj = quad.objective.quadratic.to_dict()
        for key, val in obj.items():
            m[key[0], key[1]] = val
        return m

    def compute_energy(mat, x):
        energy = 0
        for i in range(0, len(x)):
            for j in range(i, len(x)):
                if x[i] == 1 and x[j] == 1:
                    energy += mat[i][j]
        return energy

    """

    f1 = Facility(0, 2, 5, Point(1, 1))
    f2 = Facility(1, 1, 2, Point(-1, -1))

    c1 = Customer(0, 2, Point(2, 2))
    c2 = Customer(1, 5, Point(0, -1))

    facilityProblem = FacilityProblem2DPoints(2, 2, [f1, f2], [c1, c2])
    facilitySolver = VQEFacilitySolver(facilityProblem)
    facilitySolver.init_model()
    m = matrix(facilitySolver.quadratic_programm)
    print(compute_energy(m, [0,1,1,1,0,1,0,0,0,0,0]))
    
    """

    quantum_personnalized_QAOA()
    def matrix(quad: QuadraticProgram):
        """
        @param quad: a quadratic programm, must be in QUBO form
        @return: the QUBO matrix
        """
        num_var = quad.get_num_vars()
        m = np.zeros((num_var, num_var))
        obj = quad.objective.quadratic.to_dict()
        for key, val in obj.items():
            m[key[0], key[1]] = val
        return m


    def compute_energy(matrix, x):
        """
        @param matrix: a matrix of a QUBO formulation
        @param x: a binary vector
        @return: the value of the matrix for the giving vector
        """
        energy = 0
        for i in range(0, len(x)):
            for j in range(i, len(x)):
                if x[i] == 1 and x[j] == 1:
                    energy += matrix[i][j]
        return energy


    """
    # define a knapsack problem with 6 items and a capacity of 10
    max_capacity = 10

    i1 = Item(0, 4, 2)
    i2 = Item(1, 5, 2)
    i3 = Item(2, 4, 3)
    i4 = Item(3, 2, 1)
    i5 = Item(4, 5, 3)
    i6 = Item(5, 2, 1)

    # solving the knapsack problem using the VQE algorithm
    knapsackProblem = KnapsackModel([i1, i2, i3, i4, i5, i6], max_capacity)
    knapsackSolver = VQEKnapsackSolver(knapsackProblem)
    knapsackSolver.init_model()
    m = matrix(knapsackSolver.quadratic_programm)
    print(compute_energy(m, [0, 1, 0, 1, 1, 0, 0, 0, 1]))
    """

    """
    p1 = Point2D(0, 0)
    p2 = Point2D(-1, 1)
    p3 = Point2D(1, -1)
    p4 = Point2D(1, 1)
    p5 = Point2D(1, -2)

    tspProblem = TSPModel2D([p1, p2, p3, p4, p5], 5, start_index=0, end_index=4)
    tspSolver = VQETSPSolver(tspProblem)
    optimizer = BOBYQA()
    kwargs = {"optimizer": optimizer}
    tspSolver.init_model()
    m = matrix(tspSolver.quadratic_programm)
    print(compute_energy(m, [0, 0, 1, 0, 1, 0, 1, 0, 0]))
    res = tspSolver.solve(**kwargs)
    sol, _ = res.get_best_solution_fit()
    print(sol)
    print(tspProblem.satisfy(sol))
    """

    """

    data = pandas.read_csv("csv_res/Test_MIS_VQE.csv")
    nb_opti = 0
    tot = 0
    nb_not_satisfy = 0

    # relancer VQE avec 6OO iters

    for i in range(100):

        if data.iloc[i, 2] == 3:
            nb_opti += 1

        if not data.iloc[i, -1]:
            nb_not_satisfy += 1
        else:
            tot += data.iloc[i, 2]

    print(nb_opti)
    print(tot/(500 - nb_not_satisfy))
    print(nb_not_satisfy)
    """

    # we construct a little graph with 6 nodes and 8 edges
    # here the mis is {1,5,6}

    f1 = Facility(0, 2, 5, Point(1, 1))
    f2 = Facility(1, 1, 2, Point(-1, -1))

    c1 = Customer(0, 2, Point(2, 2))
    c2 = Customer(1, 5, Point(0, -1))

    facilityProblem = FacilityProblem2DPoints(2, 2, [f1, f2], [c1, c2])
    # we create a Milp Solver to create the MILP model
    milpSolver = LP_Facility_Solver(problem=facilityProblem)

    # we create the retrieve function solution, for misProblem no reconstruction of the solution is needed
    def fun(x):
        x = [i for i in x]

        facility_for_customers = [-1] * facilityProblem.customer_count

        for i in range(0, facilityProblem.facility_count):
            for j in range(0, facilityProblem.customer_count):
                if x[facilityProblem.facility_count * j + i + j] == 1:
                    facility_for_customers[j] = facilityProblem.facilities[i].index

        sol = FacilitySolution(
            facilityProblem, facility_for_customers=facility_for_customers
        )
        return sol

    # we create an instance of a GeneralQAOASolver
    misSolver = GeneralVQESolver(
        problem=facilityProblem, model=milpSolver, retrieve_solution=fun
    )
    # we initialize the solver, in fact this step transform the problem in a QUBO formulation
    misSolver.init_model()
    # we solve the mis problem
    res = misSolver.solve()

    sol, fit = res.get_best_solution_fit()
    print(sol.facility_for_customers)
    print("This solution respect all constraints : ", facilityProblem.satisfy(sol))




