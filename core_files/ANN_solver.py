# ---
# Created by aitirga at 09/10/2019
# Description: This module contains the main functions to run the ANN_solver program
# ---
import numpy as np

import core_files.ANN_core as ANN_core
import core_files.gradient_descent as GD
from core_files.constants import *


class ANN_solver(ANN_core.ANN):
    def __init__(self, dim, activation_function_list=(), eps_ini=EPS_INI, cost_function="MSE", verbose=True):
        """Sets up the ANN solver
        Creates an instance of the ANN_solver class
        :param dim: tuple containing the dimensions of the ANN, e.g. dim = [3, 10, 10, 1] sets up a 4 layer ANN with
        3 units at the input layer, 10 units on each of the two hidden layers and 1 output unit
        :param activation_function_list: tuple containing the activation functions used on every layer. Available options
        are 'sigmoid' for a sigmoidal activation function, 'linear' for a linear activation function and 'reLU' for a
        rectified linear unit activation function
        :param eps_ini: parameter that sets the range [-eps_ini, eps_ini] to initialize the weight matrix
        :param cost_function: sets up the cost function type. Available options are "MSE" for mean square error and
        "classification" for cross-entropy
        :param verbose: controls the level of verbose *not implemented
        """
        super().__init__(dim=dim, activation_function_list=activation_function_list, cost_function=cost_function,
                         eps_ini=eps_ini, verbose=verbose)
        ANN_core.ANN.build_matrices(self)
        ANN_core.ANN.randomize_weights(self)
        ANN_core.io.print_intro(self)

    def run_gradient_descent(self, alpha=1e-4, ATOL=ATOL, RTOL=RTOL, always_decrease=False,
                             n=100, plotting=True, n_plot=False, NMAX=NMAX, lambda0=0.0, input_set_type="full",
                             momentum_g=0.8, norm=False, n_batch=25, plot_contour=False, n_contour=(100, 100),
                             momentum=False, avoid_cf=False, **kwargs):
        """Runs the gradient descent algorithm
        Runs the gradient descent algorithm
        :param norm:
        :param n_batch:
        :param always_decrease:
        :param alpha: initial learning rate
        :param ATOL: absolute tolerance, if the change in cost function between consecutive steps is less than ATOL,
        the algorithm converges
        :param RTOL: relative tolerance, if the change in costa function compared to the initial one is less than RTOL,
        the algorithm converges
        :param always_decrease: set it TRUE in order to force the cost function to always decrease in value, if it does
        not do that, an error is raised
        :param n: parameter that controls the number of steps before updating the plots
        :param plotting:
        :param n_plot:
        :param NMAX: maximum number of timesteps, if the algorithm performs more timesteps, it raises an error
        :param lambda0: regularization parameter
        :param input_set_type: sets the criteria to use the input file in the calculation of the gradient descent:
        "full" to use the whole dataset, "batch" to do batches of n_batch elements and "stochastic" to use one
        input every step
        :param momentum_g: sets the momentum constant
        :param plot_contour: Set it True to dinamically plot a contour plot of the results
        :param n_contour: number of elements on the contour plot
        :param momentum: Set it True to use momentum and "Nesterov" to use the Nesterov momentum
        :param avoid_cf: Set it True to avoid the calculation of the cost function
        :return:
        """

        GD.GradientDescent.initialize(self, alpha=alpha, ATOL=ATOL, RTOL=RTOL, always_decrease=always_decrease,
                                      n=n, plotting=plotting, n_plot=n_plot, NMAX=NMAX, lambda0=lambda0,
                                      input_set_type=input_set_type,
                                      momentum_g=momentum_g, norm=norm, n_batch=n_batch, plot_contour=plot_contour,
                                      n_contour=n_contour,
                                      momentum=momentum, avoid_cf=avoid_cf)
        GD.GradientDescent.run_gradient_descent(self)

    # TODO: Take out the link of the function below from ANN_core
    def write_weights(self, filename):
        """Write computed weigths in file
        Writes the weights of the Neural Network on a text file "filename.dat" in the numpy format
        :param filename: path to the file
        """
        np.save(filename, self.Theta)
        print("The weight matrix has been properly saved in %s" % filename)

    def load_weights(self, filename):
        """Load weigths from file
        Load the weights of the Neural Network from a text file "filename.dat" in the numpy format
        :param filename: path to the file
        """
        v = np.load(filename)
        self.Theta = v
        print("The weight matrix has been properly load from %s" % filename)


if __name__ == "__main__":
    pass
