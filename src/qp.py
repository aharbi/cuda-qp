import numpy as np

from logger import SolverLogger


class QP:
    def __init__(
        self,
        c: np.ndarray,
        Q: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = False,
        solver_args=None,
    ):
        # Solver args
        default_solver_args = {
            "alpha": 0.99,
            "sigma": 0.5,
            "tol_primal": 1e-8,
            "tol_dual": 1e-8,
            "tol_opt": 1e-8,
        }

        if solver_args is None:
            solver_args = {}

        self.solver_args = {**default_solver_args, **solver_args}

        # Problem data
        self.c = c
        self.Q = Q

        self.A = A
        self.b = b

        self.m = self.A.shape[0]
        self.n = self.A.shape[1]

        assert np.linalg.matrix_rank(A) == self.m

        # Decision variables
        self.initialize_variables()

        # Barrier parameter
        self.mu = np.inner(self.x, self.s) / self.n

        # Infeasiblities
        self.update_infeasiblities()

        self.k = 0

        self.verbose = verbose

        if self.verbose:
            self.solver_logger = SolverLogger(self)

    def initialize_variables(self):
        self.x = np.ones(self.n)
        self.y = np.zeros(self.m)
        self.s = np.ones(self.n)

    def compute_objective(self):
        return np.inner(self.c, self.x) + 0.5 * np.inner(self.x, self.Q @ self.x)

    def compute_dual_objective(self):
        return np.inner(self.b, self.y) - 0.5 * np.inner(self.x, self.Q @ self.x)

    def compute_constraint_violation(self):
        return self.xi_p

    def update_infeasiblities(self):
        self.xi_p = self.b - self.A @ self.x
        self.xi_d = self.c + self.Q @ self.x - self.A.T @ self.y - self.s
        self.xi_mu = self.solver_args["sigma"] * self.mu * np.ones(self.n) - np.diag(
            self.x
        ) @ np.diag(self.s) @ np.ones(self.n)

    def compute_step_size(self, x, delta):

        alphas = -1 * x / delta

        alpha_neg = alphas[delta <= 0]

        if alpha_neg.size == 0:
            alpha = 1
        else:
            alpha = np.min(alpha_neg)

        return alpha

    def compute_direction(self, method="augmented"):

        if method == "direct":
            v = np.hstack([self.xi_p, self.xi_d, self.xi_mu])

            Z = np.block(
                [
                    [self.A, np.zeros((self.m, self.m)), np.zeros((self.m, self.n))],
                    [-self.Q, self.A.T, np.eye(self.n)],
                    [np.diag(self.s), np.zeros((self.n, self.m)), np.diag(self.x)],
                ]
            )

            delta = np.linalg.inv(Z) @ v

            delta_x = delta[: self.n]
            delta_y = delta[self.n : (self.n + self.m)]
            delta_s = delta[(self.n + self.m) :]

        elif method == "augmented":
            X_inv = np.diag(1 / self.x)
            S = np.diag(self.s)

            phi_inv = S @ X_inv

            v = np.hstack([self.xi_d - X_inv @ self.xi_mu, self.xi_p])

            Z = np.block(
                [
                    [-self.Q - phi_inv, self.A.T],
                    [self.A, np.zeros((self.m, self.m))],
                ]
            )

            delta = np.linalg.inv(Z) @ v

            delta_x = delta[: self.n]
            delta_y = delta[self.n : (self.n + self.m)]
            delta_s = X_inv @ (self.xi_mu - S @ delta_x)

        return delta_x, delta_y, delta_s

    def step(self):

        # Update barrier parameter
        self.mu = self.solver_args["sigma"] * self.mu

        # Compute the netwon direction
        delta_x, delta_y, delta_s = self.compute_direction()

        # Compute the step size
        alpha_p = self.compute_step_size(self.x, delta_x)
        alpha_d = self.compute_step_size(self.s, delta_s)

        alpha_p = self.solver_args["alpha"] * alpha_p
        alpha_d = self.solver_args["alpha"] * alpha_d

        # Update the decision variables
        self.x = self.x + alpha_p * delta_x
        self.y = self.y + alpha_d * delta_y
        self.s = self.s + alpha_d * delta_s

        # Compute the infeasiblities
        self.update_infeasiblities()

        self.k = self.k + 1

    def solve(self):

        norm_b = np.linalg.norm(self.b)
        norm_c = np.linalg.norm(self.c)

        while True:
            tol_primal_cond = np.linalg.norm(self.xi_p) / (1 + norm_b)
            tol_dual_cond = np.linalg.norm(self.xi_d) / (1 + norm_c)

            tol_opt_num = np.inner(self.x, self.s) / self.n
            tol_opt_den = 1 + np.abs(
                np.inner(self.c, self.x) + 0.5 * np.inner(self.x, self.Q @ self.x)
            )

            tol_opt_cond = tol_opt_num / tol_opt_den

            if (
                tol_primal_cond <= self.solver_args["tol_primal"]
                and tol_dual_cond <= self.solver_args["tol_dual"]
                and tol_opt_cond <= self.solver_args["tol_opt"]
            ):
                self.solver_logger.step(done=True)
                break

            self.step()

            if self.verbose:
                self.solver_logger.step(done=False)

        return self.x, self.compute_objective()


if __name__ == "__main__":
    # Generate a random non-trivial quadratic program.
    m = 300
    n = 400

    Q = np.random.randn(n, n)
    Q = Q.T @ Q
    c = np.random.randn(n)

    A = np.ones((1, n))
    b = 1

    qp = QP(c, Q, A, b, verbose=True)

    x, f = qp.solve()
