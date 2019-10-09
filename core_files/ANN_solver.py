# ---
# Created by aitirga at 09/10/2019
# Description: This module contains the main functions to run the ANN_solver program
# ---
from core_files.constants import *
import core_files.ANN_core as ANN_core

class ANN_solver(ANN_core.ANN):
    def __init__(self, dim, activation_function_list=(), eps_ini=0.12, cost_function="MSE", verbose=True):
        super().__init__(dim=dim, activation_function_list=activation_function_list, cost_function=cost_function,
                         eps_ini=eps_ini, verbose=verbose)
        ANN_core.ANN.build_matrices(self)
        ANN_core.ANN.randomize_weights(self)
        ANN_core.io.print_intro(self)


    def run_gradient_descent(self, **kwargs):
        ANN_core.ANN.gradient_descent(self, **kwargs)


if __name__ == "__main__":
    pass

