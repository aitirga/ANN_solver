# Develop ANN
from core_files.ANN_solver import ANN_solver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random as rd


def linear (x, a=1.0, b=0.0):
    return a*x + b


def quadratic(x, a=1.0, b=0.0, c=0.0):
    return a*x**2 + b*x + c


def sample_function(f, n, x_min=-1.0, x_max=1.0, n_f=100, **args):
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


def set_up_model():
    x_sample = sample_function(quadratic, 20, a=1.0, c=0.0)
    y_sample = y_from_x(x_sample, 1.0)
    x_sample1 = sample_function(quadratic, 20, a=1.0, c=0.25)
    y_sample1 = y_from_x(x_sample1, 0.0)
    x_sample2 = sample_function(quadratic, 20, a=1.0, c=-0.5)
    y_sample2 = y_from_x(x_sample2, 1.0)
    x_sample3 = sample_function(quadratic, 20, a=1.0, c=-0.75)
    y_sample3 = y_from_x(x_sample3, 0.0)

    x_sample = np.array(x_sample + x_sample1 + x_sample2 + x_sample3)
    y_sample = np.array(y_sample + y_sample1 + y_sample2 + y_sample3)

    return x_sample, y_sample


def plot_results(x_sample, y_sample, Neural_network):
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
    for i in x_sample:
        X_plot[0].append(i[0])
        X_plot[1].append(i[1])
    for i in y_sample:
        Y_plot.append(i)

    ax2.scatter(X_plot[0], X_plot[1], c=Y_plot[:])
    plt.savefig("try.png")
    plt.show()


if __name__ == '__main__':
    # This test generates a sample dataset consisting on data arranged in a parabolic shape
    # The ANN-solver is used to classify the data
    dim = [2, 15, 1]  # Dimensions of the neural network (input_layer, hidden_layer_1, ..., hidden_layer_n, output_layer)
    activation_list = ["linear", "ReLU", "sigmoid"]  #
    # x_sample, y_sample = set_up_model(quadratic, 20, a=1.0, c=0.0)
    Neural_network = ANN_solver(dim, activation_function_list=activation_list, cost_function="classification")
    x_sample, y_sample = set_up_model()
    Neural_network.create_data_vector_x(x_sample)
    Neural_network.create_data_vector_y(y_sample)
    Neural_network.normalize_x_stdmean()
    Neural_network.run_gradient_descent(alpha=0.1, ATOL=1E-20, RTOL=1E-20, always_decrease=False,
                                                    n=250, plotting=True, NMAX=5000, lambda0=0.05, input_set_type="full",
                                                    norm=False, n_batch=20, plot_contour=True)
    plot_results(x_sample, y_sample, Neural_network)  # Plot the results



