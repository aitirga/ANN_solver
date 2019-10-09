# ---
# Created by aitirga at 09/10/2019
# Description: This module contains the gradient descent class
# ---
from core_files.ANN_core import ANN
import os
import matplotlib.pyplot as plt
import time


class GradientDescent(ANN):
    def initialize(self, alpha=1e-4, ATOL=1E-3, RTOL=1E-4, AlwaysDecrease=False,
                                                    n=100, plotting=True, n_plot=False, NMAX=1E4, lambda0=0.0, input_set_type="full",
                                                    momentum_g=0.8, Norm=False, N_batch=25, plot_contour=False, n_contour=[100, 100],
                                                    momentum=False, avoid_cf=False):
        # Perform gradient descent minimization algorithm using the provided learning rate.
        # It uses a fixed number of learning steps
        # Parameters for live plotting
        self.input_set_type = input_set_type
        self.n_batch = N_batch
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
        self.tol["AlwaysDecrease"]["Value"] = AlwaysDecrease
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


        if Norm:  # Specifies if the input X vector should be normalized
            if self.normalized:
                pass
            else:
                ANN.normalize_x_stdmean(self)
        else:
            pass


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
                self.alpha = ANN.adaptive_learning_rate(self)
            ANN.compute_gradient_and_update_weights(self)
            # ANN.update_weights(self)
            if (ANN.evaluate_tolerances(self, self.ATOL, self.RTOL, self.NMAX) == True):
                break
            if self.avoid_CF:
                pass
            else:
                temp_CF = ANN.cost_function(self)
                self.CF.append(temp_CF)
            ANN.evaluate_tolerances(self, self.ATOL, self.RTOL, self.NMAX)
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
        ANN.sim_status(self)
        plt.ioff()


class io(GradientDescent):
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