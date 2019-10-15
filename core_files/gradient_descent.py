# ---
# Created by aitirga at 09/10/2019
# Description: This module contains the gradient descent class
# ---
from core_files.ANN_core import ANN
from core_files.constants import *
import os
import matplotlib.pyplot as plt
import time
import numpy as np

class GradientDescent(ANN):
    def initialize(self, alpha=1e-4, ATOL=ATOL, RTOL=RTOL, always_decrease=False,
                                                    n=100, plotting=True, n_plot=False, NMAX=NMAX, lambda0=0.0, input_set_type="full",
                                                    momentum_g=0.8, norm=False, n_batch=25, plot_contour=False, n_contour=[100, 100],
                                                    momentum=False, avoid_cf=False):
        """

        :param alpha: initial learning rate
        :param ATOL: absolute tolerance, if the change in cost function between consecutive steps is less than ATOL,
        the algorithm converges
        :param RTOL: relative tolerance, if the change in costa function compared to the initial one is less than RTOL,
        the algorithm converges
        :param AlwaysDecrease: set it TRUE in order to force the cost function to always decrease in value, if it does
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
        :param Norm: Set it True to automatically normalize the data
        :param N_batch: Number of input elements on the batch
        :param plot_contour: Set it True to dinamically plot a contour plot of the results
        :param n_contour: number of elements on the contour plot
        :param momentum: Set it True to use momentum and "Nesterov" to use the Nesterov momentum
        :param avoid_cf: Set it True to only compute the cost function at the plotting timesteps set by n
        :return:
        """
        # Perform gradient descent minimization algorithm using the provided learning rate.
        # It uses a fixed number of learning steps
        # Parameters for live plotting
        self.input_set_type = input_set_type
        self.n_batch = n_batch
        self.plotting = plotting
        self.time_start = time.time()
        self.time_0 = self.time_start
        self.lambda0 = lambda0  # Regularization term
        self.gamma = momentum_g  # The momentum term
        self.alpha = alpha  # Step size
        self.alpha0 = alpha  # Initial step size (default)
        self.ATOL = ATOL
        self.RTOL = RTOL
        self.NMAX = NMAX
        self.momentum = momentum
        self.tol["AlwaysDecrease"]["Value"] = always_decrease
        self.x_backup = self.x
        self.avoid_CF = avoid_cf
        self.plotting = plotting
        self.plot_contour = plot_contour
        self.n_contour = n_contour
        self.n = n

        if n_plot == False:
            self.n_plot = self.n
        else:
            self.n_plot = n_plot
        self.Time = []
        self.CF = []

        # Automatic tolerance stop (ATOL)
        self.nsim = 0
        # Velocities matrix for the momentum implementation


        if norm:  # Specifies if the input X vector should be normalized
            if not self.normalized:
                ANN.normalize_x_stdmean(self)


    def run_gradient_descent(self):
        line1 = []
        io.print_gradient_descent_intro(self)
        self.n_plot_list = []
        self.CF_plot_list = []
        # First timestep cf calculation
        temp_CF = ANN.cost_function(self)
        self.n_plot_list.append(self.nsim)
        self.CF_plot_list.append(temp_CF)
        line1 = ANN.plot_CF(self, line1)
        contour = []
        while 1:
            if self.adaptive_learning_rate:
                self.alpha = GradientDescent.adaptive_learning_rate(self)
            GradientDescent.compute_gradient_and_update_weights(self)
            # ANN.update_weights(self)
            if (GradientDescent.evaluate_tolerances(self, self.ATOL, self.RTOL, self.NMAX) == True):
                break
            if self.avoid_CF:
                pass
            else:
                temp_CF = ANN.cost_function(self)
                self.CF.append(temp_CF)
            GradientDescent.evaluate_tolerances(self, self.ATOL, self.RTOL, self.NMAX)
            if float(self.nsim) % self.n == 0.0:
                if self.avoid_CF:
                    temp_CF = ANN.cost_function(self)
                    self.CF.append(temp_CF)
                print("Learning step %s, cost function value: %s" % (self.nsim, temp_CF))
                ANN.print_tolerances(self)
                if self.plot_contour:
                    contour = ANN.live_plotter_contour(self, contour, self.n_contour)
                # ANN.check_gradient(self)
            self.nsim += 1
            if self.plotting:
                if float(self.nsim) % self.n == 0:
                    self.n_plot_list.append(self.nsim)
                    self.CF_plot_list.append(temp_CF)
                    line1 = ANN.plot_CF(self, line1)
        io.sim_status(self)
        plt.ioff()

    def adaptive_learning_rate(self):
        return self.dict_learning_rate[self.adaptive_learning_rate_type](self, **self.adaptive_learning_rate_args)

    def compute_gradient_and_update_weights(self):
        if not self.momentum == "nesterov":
            ANN.compute_gradient(self)
        if self.momentum == "nesterov":
            self.Theta += np.multiply(self.alpha, self.v_momentum)
            ANN.compute_gradient(self)
        if self.momentum == True or self.momentum == "yes":
            self.v_momentum = np.multiply(self.gamma, self.v_momentum) - np.multiply(self.alpha, self.D)
            self.Theta += self.v_momentum
        if self.momentum == "nesterov":
            self.v_momentum = np.multiply(self.gamma, self.v_momentum) - np.multiply(self.alpha, self.D)
            self.Theta += self.v_momentum
        if not self.momentum:
            self.Theta -= np.multiply(self.alpha, self.D)

    def evaluate_tolerances(self, ATOL=1E-2, RTOL=1E-4, NMAX=100000, AlwaysDecrease=False):
        if len(self.CF) > 1:
            self.tol["ATOL"]["Value"] = abs(self.CF[-2] - self.CF[-1])
            self.tol["RTOL"]["Value"] = abs(self.CF[-2] - self.CF[-1]) / (abs(self.CF[-1] - self.CF[0] + self.eps))
            self.tol["NMAX"]["Value"] = self.nsim
            if (self.tol["AlwaysDecrease"]["Value"] == 1.0) and (self.CF[-1] - self.CF[-2] > 0.0):
                self.tol["AlwaysDecrease"]["Status"] = True
            if self.tol["ATOL"]["Value"] < ATOL:
                self.tol["ATOL"]["Status"] = True
            if self.tol["RTOL"]["Value"] < RTOL:
                self.tol["RTOL"]["Status"] = True
            if self.tol["NMAX"]["Value"] > NMAX:
                self.tol["NMAX"]["Status"] = True
        for i in self.tol:
            if self.tol[i]["Status"]:
                return True



class io(GradientDescent):
    """Input/output control class for GradientDescent
    This class contains the functions to print output data regarding the evolution of the GradientDescent algorithm
    """
    def sim_status(self):
        reason = {}
        for i in self.tol:
            if self.tol[i]["Status"]:
                reason[i] = self.tol[i]
        print("************ Training finished ************\n")
        print("*******************************************")
        print("********** ANN training status ************")
        print("*******************************************")
        print("The training of the neural network has finished")
        print("Total training time: %s s" % (time.time() - self.time_start))
        print("Simulation steps: %s" % self.nsim)
        print("Value of the minimized Cost Function %s" % self.CF[-1])
        print("Reason for termination: %s" % reason)
        print("****************** END ********************\n")
        os.system('pause')

    def print_gradient_descent_intro(self):
        print("*******************************************")
        print("**** Gradient descent training utility ****")
        print("*******************************************")
        print("Gradient descent utility properties:")
        print("\tLearning rate: %s" % self.alpha)
        if self.plotting:
            print("\tPlotting CF evolution")
        print("Regularization term: %s" % self.lambda0)
        print("Input set for gradient calculation: %s" % self.input_set_type)
        if self.input_set_type == "batch":
            print("Batch elements: %s" % self.n_batch)
        print("Gradient acceleration: %s" % self.momentum)
        print("\tTolerances:")
        print("\t\tATOL: %s" % self.ATOL)
        print("\t\tRTOL: %s" % self.RTOL)
        print("\t\tNMAX: %s" % self.NMAX)
        print("************ Training started *************")