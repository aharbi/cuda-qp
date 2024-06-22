import numpy as np
from qp import *

if __name__ == "__main__":
    # Generate a random non-trivial quadratic program.
    import cvxpy as cp
    from datetime import datetime

    m = 300
    n = 400

    Q = np.random.randn(n, n)
    Q = Q.T @ Q
    c = np.random.randn(n)

    a1 = np.ones(n)
    a2 = np.zeros(n)
    a2[0] = 1

    A = np.vstack((a1, a2))

    b = np.array([1, 0.1])

    # Define and solve the CVXPY problem.

    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(c.T @ x + 0.5 * cp.quad_form(x, cp.psd_wrap(Q))),
        [x >= 0, A @ x == b],
    )

    start = datetime.now()
    result = prob.solve()
    print(datetime.now() - start)

    problem = QP(c, Q, A, b, verbose=True)

    start = datetime.now()
    x_qp, f = problem.solve()
    # print(datetime.now() - start)

    print(np.linalg.norm(x_qp - x.value))
    print(np.abs(f - result))
    # print(qp.compute_constraint_violation())
