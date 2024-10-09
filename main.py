import numpy as np


def simplex_solver(C, A, b, eps=1e-9):
    C = np.array(C)

    print("Optimization Problem:")
    objective_type = "max" if all(c <= 0 for c in C) else "min"
    obj_func = " + ".join(f"{C[i]} * x{i + 1}" for i in range(len(C)))
    print(f"{objective_type} z = {obj_func}")

    print("Subject to the constraints:")
    for i in range(len(A)):
        constraints = " + ".join(f"{A[i][j]} * x{j + 1}" for j in range(len(A[i])))
        print(f"{constraints} <= {b[i]}")

    num_vars = len(C)
    num_constraints = len(A)

    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))

    tableau[:num_constraints, :num_vars] = A
    tableau[:num_constraints, num_vars:num_vars + num_constraints] = np.eye(num_constraints)
    tableau[:num_constraints, -1] = b
    tableau[-1, :num_vars] = -C

    while True:
        entering_var_index = np.argmin(tableau[-1, :-1])
        entering_var_coeff = tableau[-1, entering_var_index]

        if entering_var_coeff >= -eps:
            break

        ratios = []
        for i in range(num_constraints):
            if tableau[i, entering_var_index] > eps:
                ratios.append(tableau[i, -1] / tableau[i, entering_var_index])
            else:
                ratios.append(float('inf'))

        leaving_var_index = np.argmin(ratios)

        if ratios[leaving_var_index] == float('inf'):
            return {"solver_state": "unbounded", "x_star": None, "z": None}

        pivot_value = tableau[leaving_var_index, entering_var_index]
        tableau[leaving_var_index] /= pivot_value

        for i in range(num_constraints + 1):
            if i != leaving_var_index:
                tableau[i] -= tableau[i, entering_var_index] * tableau[leaving_var_index]

    solution = np.zeros(num_vars)
    for i in range(num_vars):
        col = tableau[:, i]
        if sum(col == 1) == 1 and sum(col == 0) == num_constraints:
            row = np.where(col == 1)[0][0]
            solution[i] = tableau[row, -1]

    optimal_value = tableau[-1, -1]

    return {"solver_state": "solved", "x_star": solution, "z": optimal_value}


def run_tests():
    tests = [
        # Special Case: Degeneracy
        {
            "name": "Degenerate Case",
            "C": [5, 4],
            "A": [[1, 2], [3, 2]],
            "b": [6, 12],
        },
        # Special Case: Alternative Optima (Multiple optimal solutions)
        {
            "name": "Alternative Optima",
            "C": [6, 8],
            "A": [[1, 1], [5, 4]],
            "b": [10, 40],
        },
        # Special Case: Unbounded Solution
        {
            "name": "Unbounded Solution",
            "C": [4, 3],
            "A": [[-1, 1]],
            "b": [2],
        },
        # Standard Maximization LPP 1
        {
            "name": "Maximization LPP 1",
            "C": [3, 2],
            "A": [[2, 1], [1, 1], [1, 0]],
            "b": [10, 8, 4],
            "eps": 1e-9  # Default epsilon
        },
        # Standard Maximization LPP 2 with custom epsilon
        {
            "name": "Maximization LPP 2",
            "C": [10, 6],
            "A": [[1, 1], [2, 1], [1, 2]],
            "b": [100, 150, 120],
            "eps": 1e-6  # Custom epsilon for this test case
        },
        # Standard Maximization LPP 3 with custom epsilon
        {
            "name": "Maximization LPP 3",
            "C": [2, 5],
            "A": [[1, 2], [2, 1]],
            "b": [20, 18],
            "eps": 1e-9  # Default epsilon
        },
        # Standard Maximization LPP 4
        {
            "name": "Maximization LPP 4",
            "C": [8, 6],
            "A": [[2, 1], [1, 3]],
            "b": [16, 15],
            "eps": 1e-9  # Default epsilon
        }
    ]

    for test in tests:
        eps = test.get('eps', 1e-9)
        print("Test:", test['name'])
        result = simplex_solver(test['C'], test['A'], test['b'], eps)
        print("\nResult:")
        print("Solver State:", result['solver_state'])
        if result['solver_state'] == "solved":
            print("Optimal Decision Variables (x*):", result['x_star'])
            print("Optimal Value (z):", result['z'])
        print()


# Uncomment to run our tests
# run_tests()
