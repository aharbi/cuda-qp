import numpy as np


class SolverLogger:

    def __init__(self, solver, divider_len=63):
        self.solver = solver

        self.divider_len = divider_len
        self.divider = "-" * 63

    def print_header(self):
        header = "| {:>5} | {:>15} | {:>15} | {:>15} |"
        print(self.divider)
        print(header.format("Iter", "Primal Objective", "Dual Objective", "Gap"))
        print(self.divider)

    def print_iteration(self):
        primal_obj = self.solver.compute_objective()
        dual_obj = self.solver.compute_dual_objective()
        gap = np.abs(primal_obj - dual_obj)

        print(
            "| {:>5} | {:>15.10f} | {:>15.10f} | {:>15.10f} |".format(
                self.solver.k, primal_obj, dual_obj, gap
            )
        )

    def print_footer(self):
        print(self.divider)

    def step(self, done=False):
        if self.solver.k == 1:
            self.print_header()

        if done:
            self.print_footer()
        else:
            self.print_iteration()
