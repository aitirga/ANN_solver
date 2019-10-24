# ---
# Created by aitirga at 09/10/2019
# Description: This module contains auxiliar functions that are used in the ANN_core module
# ---
import numpy as np


def sigmoid_matrix(a, one_val=100.0, zero_val=-100.0):
    """Computes the sigmoid of a matrix
    Computes the truncated sigmoid of a given vector
    :param numpy.array a: input vector
    :param float one_val: Lower bound of the truncated sigmoid
    :param float zero_val: Upper bound of the truncated sigmoid
    :return:
    """
    # print("sigmoid: %s" % A)
    r = np.zeros_like(a)
    for j in range(0, a.shape[0]):
        val = a[j, 0]
        if val > one_val:
            sigmoid_val = 1.0
        elif val < zero_val:
            sigmoid_val = 0.0
        else:
            sigmoid_val = 1 / (1 + np.exp(-val))
        r[j, 0] = sigmoid_val
    return r


def sigmoid_matrix_derivative(vec):
    """Sigmoid matrix derivative
    Calculates the derivative of the sigmoid of a given vector.
    :param numpy.array vec: input vector
    :return: derivative of the sigmoid function
    :rtype: numpy.array
    """
    sigmoid_vector = sigmoid_matrix(vec)
    temp_array = np.multiply(sigmoid_vector, 1 - sigmoid_vector)
    return temp_array


def linear(a):
    """Linear activation function
    Computes a linear activation function
    :param a:
    :return: array
    """
    return a


def linear_derivative(vec):
    """Linear derivative activation function
    Derivative of the linear activation function
    :param vec:
    :return: array
    """
    temp_array = np.ones(shape=(vec.shape[0], 1))
    return temp_array


def relu(vec, r=0.0):
    """ReLU activation
    Computes the rectified activation unit (ReLU)
    :param numpy.array vec:
    :param float r:
    :return: reLU activated array
    :rtype: numpy.array
    """
    temp_array = np.maximum(r, vec)
    return temp_array


def relu_derivative(vec, r=0.0):
    """Derivative of the ReLU activation function
    Computes the derivative of the ReLU activation function
    :param numpy.array vec: input array or vector
    :param float r: constant that defines the position of the ReLU activation
    :return: ReLU activation function
    :rtype: numpy.array
    """
    temp_array = np.maximum(r, vec)
    temp_array[temp_array > r] = 1.0
    return temp_array


# Functions to work with internal ANN matrix
def populate_with_bias(input_vec, a):
    for i in range(1, a.shape[0]):
        a[i] = input_vec[i - 1, 0]
    return a


def populate_nodes_a_from_z(z, a):
    for i in range(0, len(z)):
        for j in range(1, z[i].shape[0]):
            a[i][j - 1, 0] = z[i][j, 0]
    return a


def populate_vector_bias_to_nobias(vec):
    a = np.ones(shape=(vec.shape[0] - 1, 1))
    for j in range(1, a.shape[0] + 1):
        a[j - 1, 0] = vec[j, 0]
    return a


def populate_vector_nobias_to_bias(vec):
    a = np.ones(shape=(vec.shape[0] + 1, 1))
    for j in range(1, a.shape[0]):
        a[j, 0] = vec[j - 1, 0]
    return a


def populate_nodes_z_from_a(a, z):
    for i in range(0, len(a)):
        for j in range(0, a[i].shape[0]):
            z[i][j + 1, 0] = a[i][j, 0]
    return z
