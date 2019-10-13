# ---
# Created by aitirga at 09/10/2019
# Description: This module contains the main functions to run the ANN_solver program
# ---
from core_files.constants import *
import core_files.ANN_core as ANN_core
import core_files.gradient_descent as GD
import numpy as np

class ANN_solver(ANN_core.ANN):
    def __init__(self, dim, activation_function_list=(), eps_ini=0.12, cost_function="MSE", verbose=True):
        super().__init__(dim=dim, activation_function_list=activation_function_list, cost_function=cost_function,
                         eps_ini=eps_ini, verbose=verbose)
        ANN_core.ANN.build_matrices(self)
        ANN_core.ANN.randomize_weights(self)
        ANN_core.io.print_intro(self)

    def run_gradient_descent(self, alpha=1e-4, ATOL=ATOL, RTOL=RTOL, AlwaysDecrease=False,
                                                    n=100, plotting=True, n_plot=False, NMAX=NMAX, lambda0=0.0, input_set_type="full",
                                                    momentum_g=0.8, Norm=False, N_batch=25, plot_contour=False, n_contour=[100, 100],
                                                    momentum=False, avoid_cf=False, **kwargs):
        # TODO: Write documentation of the gradient descent function
        """

        :param alpha:
        :param ATOL:
        :param RTOL:
        :param AlwaysDecrease:
        :param n:
        :param plotting:
        :param n_plot:
        :param NMAX:
        :param lambda0:
        :param input_set_type:
        :param momentum_g:
        :param Norm:
        :param N_batch:
        :param plot_contour:
        :param n_contour:
        :param momentum:
        :param avoid_cf:
        :param kwargs:
        :return:
        """

        GD.GradientDescent.initialize(self, alpha=alpha, ATOL=ATOL, RTOL=RTOL, AlwaysDecrease=AlwaysDecrease,
                                                    n=n, plotting=plotting, n_plot=n_plot, NMAX=NMAX, lambda0=lambda0, input_set_type=input_set_type,
                                                    momentum_g=momentum_g, Norm=Norm, N_batch=N_batch, plot_contour=plot_contour, n_contour=n_contour,
                                                    momentum=momentum, avoid_cf=avoid_cf, **kwargs)
        GD.GradientDescent.run_gradient_descent(self)
        # ANN_core.ANN.gradient_descent(self, **kwargs)

    # TODO: Take out the link of the function below from ANN_core
    def write_weights(self, filename):
        # Writes the weights of the Neural Network on a text file "filename.dat" in the numpy format
        np.save(filename, self.Theta)
        print("The weight matrix has been properly saved in %s" % filename)

    def load_weights(self, filename):
        # Writes the weights of the Neural Network on a text file "filename.dat" in the numpy format
        v = np.load(filename)
        self.Theta = v
        print("The weight matrix has been properly load from %s" % filename)


if __name__ == "__main__":
    pass

