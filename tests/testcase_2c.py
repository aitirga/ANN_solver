# Develop ANN
from core_files import ANN_core as ANN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random as rd

def linear (x, a=1.0, b=0.0):
    return a*x + b


def quadratic(x, a=1.0, b=0.0, c=0.0):
    return a*x**2 + b*x + c


def sample_function(f, n, x_min = -1.0, x_max = 1.0, n_f = 100, **args):
    x = []
    sample = rd.sample(range(0, n_f), n)
    x_space = np.linspace(x_min, x_max, n_f)#
    for i in sample:
        x.append([x_space[i], f(x_space[i], **args)])
    return x


def y_from_x(x, val):
    y = []
    for i in x:
        y.append(val)
    return y


if __name__ == '__main__':
    dim = [2, 20, 20, 1]
    activation_list = ["linear", "ReLU", "sigmoid", "sigmoid"]
    Neural_network = ANN.ANN(dim, activation_function_list=activation_list, cost_function="classification")
    X_sample = sample_function(quadratic, 20, a=1.0, c=0.0)
    Y_sample = y_from_x(X_sample, 1.0)
    X_sample1 = sample_function(quadratic, 20, a=1.0, c=0.25)
    Y_sample1 = y_from_x(X_sample, 0.0)
    X_sample2 = sample_function(quadratic, 20, a=1.0, c=-0.5)
    Y_sample2 = y_from_x(X_sample, 1.0)
    X_sample3 = sample_function(quadratic, 20, a=1.0, c=-0.75)
    Y_sample3 = y_from_x(X_sample, 0.0)

    # X_sample2 = sample_function(quadratic, 20, a=1.0, c=-0.5)
    # Y_sample2 = y_from_x(X_sample, 0.0)

    X_sample = X_sample + X_sample1 + X_sample2 + X_sample3
    Y_sample = Y_sample + Y_sample1 + Y_sample2 + Y_sample3

    # X_sample = X_sample + X_sample1 + X_sample2
    # Y_sample = Y_sample + Y_sample1 + Y_sample2

    print(X_sample)
    print(Y_sample)

    # print("Initial cost function: %s" % Neural_N.cost_function())

    Neural_network.create_data_vector_x(X_sample)
    Neural_network.create_data_vector_y(Y_sample)
    Neural_network.normalize_x_stdmean()
    # Neural_network.add_sample_data()  # Will add input X and Y to inner plots
    Neural_network.gradient_descent(alpha=0.025, ATOL=1E-20, RTOL=1E-20, AlwaysDecrease=False,
                                                    N=1000, plotting=True, NMAX=15000, lambda0=0.00025, input_set_type="batch",
                                                    Norm=False, N_batch=3, plot_contour=True)
    #Neural_network.gradient_descent_learning_auto_momentum(alpha=0.001, ATOL=1E-20, RTOL=1E-20, AlwaysDecrease=False,
    #                                                N=100, plotting=True, NMAX=1E4, lambda0=0.1,
    #                                                gamma=0.9, Norm=False)
    # Neural_network.write_weights(filename_weights)

    # print("Result of the initial case X = (0,0): %s" % Neural_network.feedforward_result_norm(
    #     np.array([[0.001], [0.00001]])))
    # Plot the feedforward result in the 2D plane
    Nx = Ny = 100
    xlist = np.linspace(-2.0, 2.0, Nx)
    ylist = np.linspace(-2.0, 2.0, Ny)
    plot_array = []
    for j in ylist:
        for i in xlist:
            plot_array.append(np.array([i, j]))
    NN_result = []
    for i in plot_array:
        a = float(Neural_network.forward_propagate(i)[0])
        NN_result.append(a)
    NN_result = np.array(NN_result).reshape([Nx, Ny])
    fig2, ax2 = plt.subplots()
    cc = ax2.contourf(xlist, ylist, NN_result)
    fig2.colorbar(cc)
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    X_plot = [[], []]
    Y_plot = []
    for i in X_sample:
        X_plot[0].append(i[0])
        X_plot[1].append(i[1])
    for i in Y_sample:
        Y_plot.append(i)

    ax2.scatter(X_plot[0], X_plot[1], c=Y_plot[:])
    plt.savefig("try.png")
    plt.show()
