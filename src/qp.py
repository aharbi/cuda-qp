import numpy as np


# TODO: Placeholder location for solver parameters
ALPHA = 0.99
SIGMA = 0.5
TOL_PRIMAL = 1e-8
TOL_DUAL = 1e-8
TOL_OPT = 1e-8


class QP:
    def __init__(
        self,
        c: np.ndarray,
        Q: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = False,
    ):

        # Problem data
        self.c = c
        self.Q = Q

        self.A = A
        self.b = b

        self.m = self.A.shape[0]
        self.n = self.A.shape[1]

        assert np.linalg.matrix_rank(A) == self.m

        # Decision variables
        self.x = None
        self.y = None
        self.s = None

        self.initialize_variables()

        # Barrier parameter
        self.mu = np.inner(self.x, self.s) / self.n

        # Infeasiblities
        self.xi_p = None
        self.xi_d = None
        self.xi_mu = None

        self.update_infeasiblities()

        self.k = 0
        self.verbose = verbose

    def initialize_variables(self):
        self.x = np.ones(self.n)
        self.y = np.zeros(self.m)
        self.s = np.ones(self.n)

    def update_infeasiblities(self):
        self.xi_p = self.b - self.A @ self.x
        self.xi_d = self.c + self.Q @ self.x - self.A.T @ self.y - self.s
        self.xi_mu = SIGMA * self.mu * np.ones(self.n) - np.diag(self.x) @ np.diag(
            self.s
        ) @ np.ones(self.n)

    def compute_step_size(self, x, delta):

        alphas = -x / delta

        alpha_neg = alphas[delta <= 0]

        if alpha_neg.size == 0:
            alpha = 1
        else:
            alpha = np.min(alpha_neg)

        return alpha

    def compute_direction(self):

        inf = np.hstack([self.xi_p, self.xi_d, self.xi_mu])

        kkt = np.block(
            [
                [self.A, np.zeros((self.m, self.m)), np.zeros((self.m, self.n))],
                [-self.Q, self.A.T, np.eye(n)],
                [np.diag(self.s), np.zeros((self.n, self.m)), np.diag(self.x)],
            ]
        )

        delta = np.linalg.inv(kkt) @ inf

        delta_x = delta[: self.n]
        delta_y = delta[self.n : (self.n + self.m)]
        delta_s = delta[(self.n + self.m) :]

        return delta_x, delta_y, delta_s

    def step(self):

        # Update barrier parameter
        self.mu = SIGMA * self.mu

        # Compute the netwon direction
        delta_x, delta_y, delta_s = self.compute_direction()

        # Compute the step size
        alpha_p = self.compute_step_size(self.x, delta_x)
        alpha_d = self.compute_step_size(self.s, delta_s)

        alpha_p = ALPHA * alpha_p
        alpha_d = ALPHA * alpha_d

        # Update the decision variables
        self.x = self.x + alpha_p * delta_x
        self.y = self.y + alpha_d * delta_y
        self.s = self.s + alpha_d * delta_s

        # Compute the infeasiblities
        self.update_infeasiblities()

        self.k = self.k + 1

    def solve(self):

        norm_b = np.linalg.norm(b)
        norm_c = np.linalg.norm(c)

        while True:
            tol_primal_cond = np.linalg.norm(self.xi_p) / (1 + norm_b)
            tol_dual_cond = np.linalg.norm(self.xi_d) / (1 + norm_c)

            tol_opt_num = np.inner(self.x, self.s) / self.n
            tol_opt_den = 1 + np.abs(
                np.inner(self.c, self.x) + 0.5 * np.inner(self.x, self.Q @ self.x)
            )

            tol_opt_cond = tol_opt_num / tol_opt_den

            if (
                tol_primal_cond <= TOL_PRIMAL
                and tol_dual_cond <= TOL_DUAL
                and tol_opt_cond <= TOL_OPT
            ):
                break

            self.step()

        return self.x


if __name__ == "__main__":
    # Generate a random non-trivial quadratic program.
    import cvxpy as cp
    from datetime import datetime

    m = 30
    n = 40

    # np.random.seed(1)

    Q = np.random.randn(n, n)
    Q = np.zeros((n, n))  # Q.T @ Q
    c = np.random.randn(n)

    A = np.ones((1, n))
    b = 1

    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(c.T @ x + 0.5 * cp.quad_form(x, cp.psd_wrap(Q))),
        [x >= 0, A @ x == b],
    )

    start = datetime.now()
    prob.solve()
    print(datetime.now() - start)

    qp = QP(c, Q, A, b)

    start = datetime.now()
    x_qp = qp.solve()
    print(datetime.now() - start)

    print(np.linalg.norm(x_qp - x.value))