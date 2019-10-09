# ---
# Created by aitirga at 09/10/2019
# Description: This module contains auxiliar functions that are used in the ANN_core module
# ---

import numpy as np


def sigmoid_matrix(A, one_val=100.0, zero_val=-100.0):
    """
    Computes the truncated sigmoid of a given vector
    :param A: input vector
    :param one_val: Lower bound of the truncated sigmoid
    :param zero_val: Upper bound of the truncated sigmoid
    :return:
    """
    # print("sigmoid: %s" % A)
    R = np.zeros_like(A)
    for j in range(0, A.shape[0]):
        val = A[j, 0]
        if val > one_val:
            sigmoid_val = 1.0
        elif val < zero_val:
            sigmoid_val = 0.0
        else:
            sigmoid_val = 1 / (1 + np.exp(-val))
        R[j, 0] = sigmoid_val
    return R


def sigmoid_matrix_derivative(vec):
    """
    Calculates the derivative of the sigmoid of a given vector.
    :param vec: input vector
    :param id:
    :return:
    """
    sigmoid_vector = sigmoid_matrix(vec)
    temp_array = np.multiply(sigmoid_vector, 1 - sigmoid_vector)
    return temp_array


def linear(A):
    return A


def linear_derivative(vec):
    temp_array = np.ones(shape=(vec.shape[0], 1))
    return temp_array


def relu(vec, r=0.0):
    temp_array = np.maximum(r, vec)
    return temp_array


def relu_derivative(vec, r=0.0):
    temp_array = np.maximum(r, vec)
    temp_array[temp_array > r] = 1.0
    return temp_array

# Functions to work with internal ANN matrix
def populate_with_bias(input, A):
    for i in range(1, A.shape[0]):
        A[i] = input[i - 1, 0]
    return A

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