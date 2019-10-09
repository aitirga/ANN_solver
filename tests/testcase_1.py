# Develop ANN
from context import ANN_solver
from core_files import ANN_core as ANN

import numpy as np
import random as rd
import matplotlib.pyplot as plt
if __name__ == '__main__':
    dim = [2, 20, 1]
    activation_list = ["linear", "sigmoid", "sigmoid"]
    Neural_network = ANN.ANN(dim, activation_function_list=activation_list, cost_function="classification")

    # Check gradient calculation
    X_sample = []
    Y_sample = []
    # print("Initial cost function: %s" % Neural_N.cost_function())
    Nx = Ny = 100
    Nsample = int(Nx*Ny/25.0)
    x = np.linspace(-2.0, 2.0, Nx)
    y = np.linspace(-2.0, 2.0, Ny)
    xx, yy = np.meshgrid(x, y)
    r = 1
    for y_i in y:
        for x_i in x:
            z = x_i**2 + y_i**2 - r
            if z <= 0.0:
                Y_sample.append(1.0)
            else:
                Y_sample.append(0.0)
    sample_pool = rd.sample(range(len(Y_sample)), Nsample)
    Y_sample_good =[]
    X_sample_good = []
    # for i in range(len(Y_sample)):
    #     iy = np.floor(i/Ny)
    #     ix = i - iy*Nx
    #     Y_sample_good.append(Y_sample[i])
    #     print(ix, iy)
    for i in sample_pool:
        iy = int(np.floor(i/Ny))
        ix = int(i - iy*Nx)
        Y_sample_good.append(Y_sample[i])
        X_sample_good.append([xx[ix, iy], yy[ix, iy]])

    fig, ax = plt.subplots(figsize=[5, 5])
    x_plot_ini = []
    y_plot_ini = []
    for i in X_sample_good:
        x_plot_ini.append(i[0])
        y_plot_ini.append(i[1])
    ax.scatter(x_plot_ini, y_plot_ini, Y_sample_good)
    plt.show()
    Neural_network.create_data_vector_x(X_sample_good)
    Neural_network.create_data_vector_y(Y_sample_good)
    # Neural_network.normalize_X_stdmean()
    # Neural_network.check_gradient3()
    # print(Neural_network.y)

    Neural_network.gradient_descent(alpha=0.01, ATOL=1E-10, RTOL=1E-10, AlwaysDecrease=False, N=50, plotting=True,
                                    NMAX=5000, lambda0=0.0, input_set_type="full", Norm=False)
    # Neural_network.gradient_descent(alpha=0.01, ATOL=1E-20, RTOL=1E-20,
    #                                                 N=500, plotting=True, NMAX=3000, lambda0=0.0, input_set_type="batch",
    #                                                 momentum=True, momentum_g=0.9, N_batch=50, plot_contour=False)
    plot_array = []
    for i in x:
        for j in y:
            plot_array.append(np.array([i, j]))
    NN_result = []
    for i in plot_array:
        a = float(Neural_network.forward_propagate(i)[0])
        NN_result.append(a)
    NN_result = np.array(NN_result).reshape([Nx, Ny])
    fig2, ax2 = plt.subplots()
    contour_plot = ax2.pcolormesh(x, y, NN_result)
    fig2.colorbar(contour_plot)
    plt.savefig("try.png")
    plt.show()

