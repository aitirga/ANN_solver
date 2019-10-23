# Develop ANN
from core_files.ANN_solver import ANN_solver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def set_up_model():
    x_sample = []
    y_sample = []
    x_sample.append(np.array([[1.0], [1.0]]))
    y_sample.append(np.array([[0.0]]))
    x_sample.append(np.array([[0.0], [1.0]]))
    y_sample.append(np.array([[0.0]]))
    x_sample.append(np.array([[2.0], [0.0]]))
    y_sample.append(np.array([[1.5]]))
    x_sample.append(np.array([[0.0], [0.0]]))
    y_sample.append(np.array([[0.5]]))
    x_sample.append(np.array([[1.5], [1.5]]))
    y_sample.append(np.array([[0.5]]))
    x_sample.append(np.array([[-1.0], [-1.0]]))
    y_sample.append(np.array([[0.5]]))
    x_sample.append(np.array([[-1.1], [-1.1]]))
    y_sample.append(np.array([[0.5]]))
    x_sample.append(np.array([[-0.9], [-0.9]]))
    y_sample.append(np.array([[0.5]]))
    x_sample.append(np.array([[1.0], [-1.5]]))
    y_sample.append(np.array([[1.5]]))
    return x_sample, y_sample


def plot_results(x_sample, y_sample, Neural_network):
    nx = ny = 100
    xlist = np.linspace(-2.0, 2.0, nx)
    ylist = np.linspace(-2.0, 2.0, ny)
    z = xlist*ylist
    plot_array = []
    for j in ylist:
        for i in xlist:
            plot_array.append(np.array([i, j]))
    NN_result = []
    for i in plot_array:
        a = float(Neural_network.forward_propagate(i)[0])
        NN_result.append(a)
    NN_result = np.array(NN_result).reshape([nx, ny])
    fig2, ax2 = plt.subplots()
    cc = ax2.contourf(xlist, ylist, NN_result)
    fig2.colorbar(cc)
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    x_plot = [[], []]
    y_plot = []
    for i in x_sample:
        x_plot[0].append(i[0])
        x_plot[1].append(i[1])
    for i in y_sample:
        y_plot.append(i[0])
    ax2.scatter(x_plot[0], x_plot[1], c=y_plot[:])
    plt.show()


if __name__ == '__main__':
    dim = [2, 10, 2, 1]  # Dimensions of the neural network (input_layer, hidden_layer_1, ..., hidden_layer_n, output_layer)
    activation_list = ["linear", "ReLU", "sigmoid", "linear"]  # List of activation functions
    Neural_network = ANN_solver(dim, activation_function_list=activation_list, cost_function="MSE")
    # Set-up the model
    x_sample, y_sample = set_up_model()

    # print("Initial cost function: %s" % Neural_N.cost_function())
    Neural_network.create_data_vector_x(x_sample)
    Neural_network.create_data_vector_y(y_sample)
    Neural_network.normalize_x_stdmean()
    Neural_network.run_gradient_descent(alpha=0.1, ATOL=1E-20, RTOL=1E-20, AlwaysDecrease=False,
                                                    n=500, plotting=True, NMAX=4000, lambda0=0.0025, input_set_type="full",
                                                    momentum_g=0.9, Norm=False, N_batch=4, plot_contour=True)
    # Plot and compare the results
    plot_results(x_sample, y_sample, Neural_network)
