###
# Created on 13 nov. 2017
#
# @author: aitirga
###
import random as rd

import matplotlib.pyplot as plt
import numpy as np

import ann_solver.auxiliar_functions as aux  # Auxiliar functions
# from ann_solver.gradient_descent import GradientDescent
from ann_solver.constants import *  # Parameters and constants


def print_nodes(x, nodes):
    print("*****************************")
    print("Nodes information:")
    print(x)
    for i in range(1, len(nodes)):
        print(nodes[i])
    print("")


def convert_vector(vec):
    """
    This function takes the input vector x and transforms it into the format used inside the class
    :param np.ndarray vec: input vector
    :return: vector in the correct format
    :rtype:np.ndarray
    """
    try:
        if len(vec.shape) == 2:
            return vec
        elif len(vec.shape) == 1:
            input_vec = np.ones(shape=[vec.shape[0], 1])
            for idx, item in enumerate(vec):
                input_vec[idx, 0] = item
            return input_vec
        elif len(vec.shape) == 0:
            input_vec = np.ones(shape=[1, 1])
            input_vec[0, 0] = vec
            return input_vec

    except AttributeError:
        try:
            input_vec = np.ones(shape=[len(vec), 1])
            for idx, item in enumerate(vec):
                input_vec[idx, 0] = item
            return input_vec
        except TypeError:  # Assuming just one element on each vector
            input_vec = np.ones(shape=[1, 1])
            input_vec[0, 0] = vec
            return input_vec


class ANN:
    # from ann_solver.io_module import io
    # This is the Neural Network class builder.
    # Input parameters are:
    # 1. dim vector, containing the inner structure of the NN layers

    def __init__(self, dim, activation_function_list=None, eps_ini=EPS_INI, cost_function="MSE", verbose=True):
        if activation_function_list is None:
            activation_function_list = []
        self.dim = dim
        self.activation_function_list = activation_function_list
        if not self.activation_function_list:
            self.activation_function_list.append("linear")  # "l" stands for linear activation
            for i in range(1, len(dim)):
                self.activation_function_list.append("sigmoid")  # "s" stands for sigmoid activation

        self.eps_ini = eps_ini
        self.eps = 1E-200
        self.normalized = False  # Boolean that tracks if input data should be normalized
        self.cost_function_type = cost_function
        self.input_set_type = "full"  # By default, all the elements in X are considered
        self.Ndim = len(dim)
        self.lambda0 = 0.0
        self.use_external_weights = False
        self.add_sample_data_to_contour = False
        self.x = []
        self.y = []
        self.tol = {"ATOL": {"Status": False, "Value": 1E50},
                    "RTOL": {"Status": False, "Value": 1E50},
                    "NMAX": {"Status": False, "Value": 1},
                    "AlwaysDecrease": {"Status": False, "Value": 0.0}}
        self.S_cross_validation = False
        self.max_sigmoid_derivative = 1E4
        self.adaptive_learning_rate = False
        self.multiprocessing = False
        self.avoid_CF = False

        # This dictionary stores the activation functions that have been implemented.
        self.dict_learning_rate = {"constant": ANN.constant_learning_rate,
                                   "linear-decay": ANN.linear_decay_learning_rate,
                                   "quadratic-decay": ANN.quadratic_decay_learning_rate,
                                   "exponential-decay": ANN.exponential_decay_learning_rate}
        self.dict_activation_function = {"linear": aux.linear,
                                         "sigmoid": aux.sigmoid_matrix,
                                         "ReLU": aux.relu}
        self.dict_activation_function_derivative = {"linear": aux.linear_derivative,
                                                    "sigmoid": aux.sigmoid_matrix_derivative,
                                                    "ReLU": aux.relu_derivative}
        # self.dict_cost_function = {"MSE": ANN.mse_cost_function, "classification": ANN.classification_cost_function}
        # This distionary stores the cost functions that have been implemented
        self.dict_cost_function_derivative = {"MSE": ANN.mse_cost_function_derivative,
                                              "classification": ANN.classification_cost_function_derivative}
        self.dict_cost_function = {"MSE": ANN.mse_cost_function,
                                   "classification": ANN.classification_cost_function}
        self.dict_input_set = {"full": ANN.full_input_set,
                               "stochastic": ANN.stochastic_input_set,
                               "batch": ANN.batch_input_set
                               }
        # ANN.build_matrices(self)
        # ANN.randomize_weights(self)
        # io.print_intro(self)
        # ANN.handle_exceptions(self)

    def use_multiprocessing(self, processes=8):
        self.multiprocessing = True
        self.n_processes = processes
        # from pathos.multiprocessing import ProcessingPool
        # import pathos.multiprocessing as mp
        # from joblib import Parallel, delayed

    @staticmethod
    def read_pd_array(v):
        """Reads array in pandas format
        Automatically reads an input array given by pandas package
        :param np.ndarray v:
        :return: array in correct format
        """
        full_v = []
        for i in range(v.shape[0]):
            temp_v = np.array(v.iloc[i, :].values)
            temp_v = convert_vector(temp_v)
            full_v.append(temp_v)
        return full_v

    def create_data_vector_x(self, x):
        """Sets input vector in the ANN module
        Given an array, it stores the input data correctly inside the ANN module
        :param np.ndarray x: input array (all the samples)
        :return:
        """
        print("Setting up input vectors of the training set")
        if str(type(x)) == "<class 'pandas.core.frame.DataFrame'>":
            x_temp = ANN.read_pd_array(x)
            self.x = x_temp
            return 1
        x_temp = []
        for i in x:
            x_temp.append(convert_vector(i))
        self.x = x_temp
        print("\tInput array set up, number of elements: %s" % len(self.x))
        return 1

    def add_data_element_x(self, x):
        """
        Adds an input data element to input vector X
        :param x:
        :return:
        """
        temp_numpy = np.zeros(shape=(len(x), 1))
        for i in range(0, len(x)):
            temp_numpy[i, 0] = x[i]
        self.x.append(temp_numpy)

    def create_data_vector_y(self, y):
        """
        Given an array, it stores the output data correctly inside the ANN module
        :param np.ndarray y: output array (all the samples)
        :return:
        """
        print("Setting up output vectors of the training set")
        if str(type(y)) == "<class 'pandas.core.frame.DataFrame'>":
            x_temp = ANN.read_pd_array(y)
            self.x = x_temp
            return 1
        y_temp = []
        for i in y:
            y_temp.append(convert_vector(i))
        self.y = y_temp
        print("Output array set up, number of elements: %s" % len(self.x))

    def add_data_element_y(self, y):
        temp_numpy = np.zeros(shape=(len(y), 1))
        for i in range(0, len(y)):
            temp_numpy[i, 0] = y[i]
        self.y.append(temp_numpy)

    def add_sample_data(self):
        self.add_sample_data_to_contour = True

    def normalize_v(self, v):
        """Normalize vector using max/min criteria
        Normalizes the input vector using the max/min criteria
        v_norm = (v_input - min_input) / (max_input - min_input)

        :param v: input to normalize
        :return: normalized array
        :rtype: np.ndarray
        """
        temp_v = np.zeros_like(self.x[0])
        for i in range(0, self.x[0].shape[0]):
            temp_v[i, 0] = 1 / (self.xmax[i] - self.xmin[i]) * (np.float32(v[i]) - self.xmin[i])
        return temp_v

    def normalize_v_stdmean(self, v):
        """Normalize vector using std/mean criteria
        Normalizes the input vector using the std/mean criteria
        v_norm = (v_input - mean_input) / (std_input)
        :param v: input to normalize
        :return: normalized array
        :rtype: np.ndarray
        """
        temp_v = np.zeros_like(self.x[0])
        for i in range(0, self.x[0].shape[0]):
            if self.xstd[i] == 0.0:
                continue
            temp_v[i, 0] = (np.float32(v[i]) - self.xmean[i]) / self.xstd[i]
        return temp_v

    def normalize_x(self):
        """Normalize input set using max/min criteria
        Normalizes the whole input set using the max/min criteria
        v_norm = (v_input - min_input) / (max_input - min_input)
        :return:
        """
        self.xmin = np.ones(shape=(self.x[0].shape[0]))
        self.xmax = np.ones(shape=(self.x[0].shape[0]))
        print(self.xmin)
        for i in range(0, self.x[0].shape[0]):
            self.xmin[i] = self.x[0][i, 0]
            self.xmax[i] = self.x[0][i, 0]

        for i in range(0, len(self.x)):
            # print(self.x[i])
            for j in range(0, self.x[i].shape[0]):
                if self.x[i][j, 0] <= self.xmin[j]:
                    self.xmin[j] = self.x[i][j, 0]
                if self.x[i][j, 0] >= self.xmax[j]:
                    self.xmax[j] = self.x[i][j, 0]
        self.xnorm = []
        for i in range(0, len(self.x)):
            temp_array = np.zeros(shape=(self.x[i].shape[0], 1))
            for j in range(0, self.x[i].shape[0]):
                temp_array[j, 0] = 1 / (self.xmax[j] - self.xmin[j]) * (self.x[i][j, 0] - self.xmin[j])
            self.xnorm.append(np.array(temp_array))

    def normalize_x_stdmean(self):
        """Normalize input set using std/mean criteria
        Automatically normalizes the input vector X, following the equation:
        X_norm_i = (X_i - X_mean)/X_std
        :return:
        """
        print("STD-MEAN normalization is activated")
        self.x_backup = self.x
        self.xmean = np.mean(self.x, axis=0)
        self.xstd = np.std(self.x, axis=0)
        self.xnorm = []
        self.normalized = True
        for i in range(0, len(self.x)):
            temp_array = np.zeros(shape=(self.x[i].shape[0], 1))
            for j in range(0, self.x[i].shape[0]):
                if self.xstd[j] == 0.0:
                    continue
                temp_array[j, 0] = (self.x[i][j, 0] - self.xmean[j]) / self.xstd[j]
            self.xnorm.append(np.array(temp_array))
        self.x = self.xnorm
        print("Input array has been normalized")
        # print(self.xnorm)

    def initialize_cross_validation(self, r_train=0.6, r_cv=0.2, r_test=0.2):
        """
        This function divides the sample dataset into three different datasets in order to use cross-validation and test verification
        :param float r_train: Ratio of the sample space to be used as the training dataset
        :param float r_cv: Ratio of the sample space to be used as the cross validation dataset
        :param float r_test: Ratio of the sample space to be used as the test dataset
        :return:
        """
        self.xtrain = []
        self.xcv = []
        self.xtest = []
        n_train = np.floor(r_train * len(self.x))
        n_cv = np.floor(r_cv * len(self.x))
        n_test = np.floor(r_test * len(self.x))
        whole_list = range(0, len(self.x))

        list_train = rd.choice(whole_list, n_train)
        list_cv = rd.choice(whole_list, n_cv)
        list_test = rd.choice(whole_list, n_test)
        print(list_train)

    def build_matrices(self):
        """ Builds all necessary matrices that will be used
        :return: self instance with the structures of the matrix set up
        :rtype: self
        """
        #
        self.Theta = []  # This is the weight tensor (connection matrix)
        self.ThetaPlus = []  # This is the weight tensor (for gradient checking)
        self.ThetaMinus = []  # This is the weight tensor (for gradient checking)
        self.Delta = []  # This is the error tensor
        self.D = []  # This is the gradient tensor (connection matrix)
        self.Dcheck = []  # This is the gradient tensor (connection matrix)
        self.nodes_z = []  # This is the nodes matrix used in forward propagation
        self.nodes_a = []  # This is the nodes matrix used in forward propagation (without the bias term)
        self.nodes_delta = []  # This is the nodes matrix used in forward propagation (without the bias term)
        self.v_momentum = []  # This is the matrix used for the momentum implementation of the gradient descent
        for i in range(0, len(self.dim) - 1):
            self.Theta.append(np.zeros(shape=(self.dim[i + 1], self.dim[i] + 1)))  # Accounted for the bias term
            self.v_momentum.append(np.zeros(shape=(self.dim[i + 1], self.dim[i] + 1)))  # Accounted for the bias term
            self.ThetaPlus.append(np.zeros(shape=(self.dim[i + 1], self.dim[i] + 1)))  # Accounted for the bias term
            self.ThetaMinus.append(np.zeros(shape=(self.dim[i + 1], self.dim[i] + 1)))  # Accounted for the bias term
            self.Delta.append(np.zeros(shape=(self.dim[i + 1], self.dim[i] + 1)))  # Accounted for the bias term
            self.D.append(np.zeros(shape=(self.dim[i + 1], self.dim[i] + 1)))  # Accounted for the bias term
            self.Dcheck.append(np.zeros(shape=(self.dim[i + 1], self.dim[i] + 1)))  # Accounted for the bias term
        for i in range(0, len(self.dim)):
            self.nodes_z.append(np.ones(shape=(self.dim[i] + 1, 1)))
            self.nodes_a.append(np.ones(shape=(self.dim[i], 1)))
            self.nodes_delta.append(np.zeros(shape=(self.dim[i], 1)))

    def randomize_matrix(self, a):
        """Randomizes initial matrix
        Randomizes initial matrix using the parameter eps_ini U_ij = U_ij + random([-eps_ini, eps_ini])
        :param np.ndarray a: input array
        :return: randomized array
        :rtype: np.ndarray
        """
        shape_m = a.shape
        rand_m = np.zeros(shape=shape_m)
        for i in range(0, shape_m[0]):
            for j in range(0, shape_m[1]):
                val = rd.uniform(-self.eps_ini, self.eps_ini)
                rand_m[i, j] = val
        return rand_m

    def randomize_weights(self):
        """Randomizes the weights of the ANN solver
        Randomizes the weigths of the ANN solver
        :return: randomized weigth matrix
        :rtype: np.ndarray
        """
        id_number = 0
        for i in self.Theta:
            i = ANN.randomize_matrix(self, i)
            self.Theta[id_number] = i
            id_number += 1

    def initialize_tensor(self, a):
        """Initializes a three dimensional tensor
        Initilizes a three dimensional tensor to zero
        :param a:
        :return: Initialized tensor
        """
        for k in range(0, len(a)):
            for i in range(0, self.dim[k + 1]):
                for j in range(0, self.dim[k] + 1):
                    a[k][i, j] = 0.0
        return a

    def apply_activation_function(self, vec, idx):
        """Sets the activation function dictionary
        Function that applies an activation function defined at "dict_activation_function" dictionary to the vector vec.
        :param vec: target of the activation function
        :param idx: index that targets the type of activation function used in layer l
        :return:
        """
        function_key = self.activation_function_list[idx]
        return self.dict_activation_function[function_key](vec)

    def apply_activation_function_derivative(self, vec, idx):
        """ Sets the activation function derivative dictionary
        Function that applies the derivative of an activation function defined at "dict_activation_function_derivative"
        dictionary to the vector vec
        :param vec:
        :param idx:
        :return:
        """
        function_key = self.activation_function_list[idx]
        return self.dict_activation_function_derivative[function_key](vec)

    # Input sets
    def get_input_set(self):
        """ Gets input set from dict_input_set dictionary
        Gets input set from dict_input_set dictionary
        :return: input set
        """
        return self.dict_input_set[self.input_set_type](self)

    def full_input_set(self):
        """Outputs the "full" input set
        Outputs the "full" input set
        :return: Whole input set
        """
        set_x = range(len(self.x))
        return set_x

    def stochastic_input_set(self):
        """Outputs one element of the input set
        Outputs one elements of the input set
        :return: one element of the input set
        """
        set_x = rd.choice(range(len(self.x)))
        return [set_x]

    def batch_input_set(self):
        """Outputs a batch of input elements
        Creates a batch of n_batch elements from the input set
        :return: batch of input set
        """
        set_x = rd.sample(range(0, len(self.x)), self.n_batch)
        return set_x

    def forward_propagation(self, x):
        """Forward propagates the input set
        This functions performs the forwards propagation algorithm using the actual weigths of the ANN module
        and an input vector X.
        :param array x: input array
        :return: output layer after forward propagating
        """
        # Populate the nodes_z matrix with the input set vector X, taking into account not to overwrite the bias term
        # Apply the first activation function
        self.nodes_z[0] = aux.populate_vector_nobias_to_bias(x)
        temp_vec = ANN.apply_activation_function(self, self.nodes_z[0], 0)
        self.nodes_a[0] = aux.populate_vector_bias_to_nobias(temp_vec)
        for i in range(0, self.Ndim - 1):  # We have one less weight matrix
            temp_vec = np.dot(self.Theta[i], aux.populate_vector_nobias_to_bias(self.nodes_a[i]))
            self.nodes_z[i + 1] = aux.populate_vector_nobias_to_bias(temp_vec)
            temp_vec = ANN.apply_activation_function(self, self.nodes_z[i + 1], i + 1)
            self.nodes_a[i + 1] = aux.populate_vector_bias_to_nobias(temp_vec)
        return self.nodes_a

    def forward_propagate(self, x):
        if str(type(x)) == "<class 'pandas.core.frame.DataFrame'>":
            x_temp = ANN.read_pd_array(x)
        else:
            x_temp = convert_vector(x)
        if self.normalized:
            x_temp = ANN.normalize_v_stdmean(self, x_temp)
        forward_propagate = ANN.forward_propagation(self, x_temp)
        return forward_propagate[-1]

    def forward_propagation_weight(self, x, theta):
        """Forward propagates the input set given a weight matrix
        This functions performs the forwards propagation algorithm using a given weight matrix and an input vector X.
        :param array x: input array
        :param array theta: weight matrix
        :return: output layer after forward propagating
        """
        # Populate the nodes_z matrix with the input set vector X, taking into account not to overwrite the bias term
        # Apply the first activation function
        self.nodes_z[0] = ANN.populate_vector_nobias_to_bias(x)
        temp_vec = ANN.apply_activation_function(self, self.nodes_z[0], 0)
        self.nodes_a[0] = ANN.populate_vector_bias_to_nobias(temp_vec)
        for i in range(0, self.Ndim - 1):  # We have one less weight matrix
            temp_vec = np.dot(theta[i], ANN.populate_vector_nobias_to_bias(self.nodes_a[i]))
            self.nodes_z[i + 1] = ANN.populate_vector_nobias_to_bias(temp_vec)
            temp_vec = ANN.apply_activation_function(self, self.nodes_z[i + 1], i + 1)
            self.nodes_a[i + 1] = ANN.populate_vector_bias_to_nobias(temp_vec)
        return 1

    def mse_cost_function(self):
        """Mean square error cost function
        Computes the mean square error cost function
        :return: value of the mean square error cost function
        """
        ff_result = self.nodes_a[-1]
        for i in range(0, ff_result.shape[0]):
            self.val_cf += 1 / 2 * (self.y_sample_cf[i] - self.nodes_a[-1][i]) ** 2  # Min square
        return self.val_cf

    def mse_cost_function_derivative(self):
        """Derivative of the mean square error cost function
        Computes the derivative of the mean square error cost function
        :return:
        """
        return self.nodes_a[-1] - self.y_test

    def classification_cost_function(self):
        """Cross-entropy cost function
        Computes the cross-entropy cost function
        :return: value of the cost function
        """
        ff_result = self.nodes_a[-1]
        for i in range(0, ff_result.shape[0]):
            self.val_cf += -(self.y_sample_cf[i] * np.log(ff_result[i] + self.eps) + (1 - self.y_sample_cf[i]) * np.log(
                1 - ff_result[i] + + self.eps))  # Classification
        return self.val_cf

    def classification_cost_function_derivative(self):
        """Derivative of the cross-entropy cost function
        Computes the derivative of the cross-entropy cost function
        :return: derivative of the cross-entropy cost function
        """
        temp_array = self.nodes_a[-1] - self.y_test
        temp_array2 = np.multiply(self.nodes_a[-1], 1 - self.nodes_a[-1]) + self.eps
        # if temp_array2 < self.eps:
        #    temp_array2 = self.eps
        temp_array2 = 1 / temp_array2
        return np.multiply(temp_array, temp_array2)

    def cost_function_value(self):
        return self.dict_cost_function[self.cost_function_type](self)

    def cost_function(self):
        """Computes the cost function
        Based on the current set-up of the model, it computes the cost function
        using the sample set that is defined and the type of cost function
        :return: cost function value
        """
        # Iteration over the training set
        input_set = ANN.get_input_set(self)
        n_x = len(input_set)
        self.val_cf = 0.0
        for i in input_set:
            self.x_sample_cf = self.x[i]
            self.y_sample_cf = self.y[i]
            # Run first the forward_propagation algorithm to calculate the a values
            ANN.forward_propagation(self, self.x_sample_cf)
            self.val_cf = ANN.cost_function_value(self)  # Computes the cost function
        self.val_cf = self.val_cf / n_x
        # Compute the regularization term
        reg = 0.0
        for i in range(0, self.Ndim - 1):
            for j in range(1, self.dim[i] + 1):
                for k in range(0, self.dim[i + 1]):
                    reg += self.Theta[i][k, j] ** 2
        reg *= self.lambda0 / (2 * n_x)
        cost = float(self.val_cf + reg)
        self.val_cf = cost
        return cost

    def cost_function_weight(self, x, y, theta):
        """Computes the cost function given based on an input set, output set and weigth matrix
        :param np.ndarray x: input set
        :param np.ndarray y: output set
        :param np.ndarray theta: weight matrix
        :return: cost function
        """
        # Iteration over the training set
        n_x = len(x)
        self.val_cf = 0.0
        for i in range(len(x)):
            self.x_sample_cf = x[i]
            self.y_sample_cf = y[i]
            # Run first the forward_propagation algorithm to calculate the a values
            ANN.forward_propagation_weight(self, self.x_sample_cf, theta)
            ANN.cost_function_value(self)  # Computes the cost function
        self.val_cf = self.val_cf / n_x
        # Compute the regularization term
        reg = 0.0
        for i in range(0, self.Ndim - 1):
            for j in range(1, self.dim[i] + 1):
                for k in range(0, self.dim[i + 1]):
                    reg += theta[i][k, j] ** 2
        reg *= self.lambda0 / (2 * n_x)
        cost = float(self.val_cf + reg)
        self.val_cf = cost
        return self.val_cf

    def individual_gradient(self, r):
        """Computes gradient of the cost function of a sample with respect to the elements of the weigth matrix
        This function calculates the gradient of the cost function of a single sample with respect
        to the elements of the weigth matrix using the backpropagation algorithm
        :param r: sample to compute the gradient
        """
        self.x_test = self.x[r]
        self.y_test = self.y[r]
        ANN.forward_propagation(self, self.x_test)
        self.nodes_delta[-1] = self.dict_cost_function_derivative[self.cost_function_type](
            self)  # For sigmoid activation in first layer
        try_array = ANN.apply_activation_function_derivative(self,
                                                             aux.populate_vector_bias_to_nobias(self.nodes_z[-1]),
                                                             -1)
        try_array = np.multiply(self.nodes_delta[-1], try_array)
        self.nodes_delta[-1] = try_array

        # Backpropagate the errors
        for i in range(len(self.dim) - 1, 1, -1):
            vec = np.dot(self.Theta[i - 1].transpose(), self.nodes_delta[i])
            activation_function_derivative = ANN.apply_activation_function_derivative(self, self.nodes_z[i - 1], i - 1)
            vec = np.multiply(vec, activation_function_derivative)
            self.nodes_delta[i - 1] = aux.populate_vector_bias_to_nobias(vec)

        # Compute the accumulative Delta
        for k in range(0, len(self.dim) - 1):
            temp_tensor = np.outer(self.nodes_delta[k + 1],
                                   aux.populate_vector_nobias_to_bias(self.nodes_a[k]).transpose())
            self.Delta[k] += temp_tensor

    def compute_gradient(self):
        """Computes the partial derivatives of the cost function with respect to the weigths for the input sample
        Using the backpropagation algorithm, this function calculates the gradient of the cost function with
        respect to the elements of the weight matrix
        :return: gradient array
        """
        # Computes the partial derivatives of the cost function in respect to the weigths for a given training sample (Xi, Yi)
        # First we need to run the forward propagation algorithm
        # Reinitialize Delta function (?)
        # Compute the last error function (i.e. the difference between the training set Y and the computed a)
        self.D = ANN.initialize_tensor(self, self.D)
        self.Delta = ANN.initialize_tensor(self, self.Delta)
        input_set = ANN.get_input_set(self)
        n_x = len(input_set)
        if self.multiprocessing:
            pass
            # Parallel(n_jobs=self.n_processes)(delayed(ANN.individual_gradient)(self, i) for i in input_set)
            # pool = mp.Pool(self.n_processes)
            # pool.map(unwrap_self_f, zip([self] * len(input_set), input_set))
        if not self.multiprocessing:
            for r in input_set:
                ANN.individual_gradient(self, r)
        # Compute the Derivative term:
        for k in range(0, len(self.dim) - 1):
            for j in range(0, self.dim[k] + 1):
                for i in range(0, self.dim[k + 1]):
                    if j != 0:
                        self.D[k][i, j] = 1 / n_x * (self.Delta[k][i, j] + self.lambda0 * self.Theta[k][i, j])
                    if j == 0:
                        self.D[k][i, j] = 1 / n_x * (self.Delta[k][i, j])
        return self.D

    def check_gradient(self, eps=1E-5):
        """Checks the gradient computed using the gradient descent algorithm with the one computed using finite differences
        This function checks wheter the gradient array computed with the backpropagation algorithm is correct. To check
        it, it computes the gradient in two ways: using the backprogation algorithm and using a simple finite difference
        method. Finally, a ratio between the two gradients is outputted
        :param float eps: parameter to compute the finite difference
        :return: ratio of the gradients
        """
        # print("ratio: %s" % self.Dcheck)
        ANN.compute_gradient(self)
        n_x = len(self.x)
        theta_plus = []
        theta_minus = []
        for i in range(0, len(self.dim) - 1):
            theta_plus.append(np.zeros(shape=(self.dim[i + 1], self.dim[i] + 1)))  # Accounted for the bias term
            theta_minus.append(np.zeros(shape=(self.dim[i + 1], self.dim[i] + 1)))  # Accounted for the bias term
        for k in range(0, len(self.dim) - 1):
            for j in range(0, self.dim[k] + 1):
                for i in range(0, self.dim[k + 1]):
                    theta_plus[k][i, j] = self.Theta[k][i, j]
                    theta_minus[k][i, j] = self.Theta[k][i, j]
        # print("thetaPlus: %s" %ThetaPlus)

        for k in range(0, len(self.dim) - 1):
            for j in range(0, self.dim[k] + 1):
                for i in range(0, self.dim[k + 1]):
                    # ThetaMinus[k][i,j] -= eps
                    theta_minus[k][i, j] -= eps
                    theta_plus[k][i, j] += eps
                    # print("Theta_plus %s" % ThetaPlus)
                    # print("Theta_minus %s" % ThetaMinus)
                    theta_minus_cost = ANN.cost_function_weight(self, self.x, self.y, theta_minus)
                    theta_plus_cost = ANN.cost_function_weight(self, self.x, self.y, theta_plus)
                    # ThetaMinus[k][i,j] += eps
                    theta_plus[k][i, j] -= eps
                    theta_minus[k][i, j] += eps
                    # print("Checking costs: %s %s" % (ThetaMinus_cost, ThetaPlus_cost))
                    # print("Derivative: %s" % ((ThetaPlus_cost - ThetaMinus_cost)/(2*eps)))
                    self.Dcheck[k][i, j] = (theta_plus_cost - theta_minus_cost) / (2 * eps)
        # print(self.Dcheck)
        for k in range(0, len(self.dim) - 1):
            for j in range(0, self.dim[k] + 1):
                for i in range(0, self.dim[k + 1]):
                    self.Dcheck[k][i, j] = self.Dcheck[k][i, j] / self.D[k][i, j]
        print("ratio: %s" % self.Dcheck)
        # print(self.nodes_z[1])
        # print(ANN.relu(self.nodes_z[1]))
        # print(ANN.relu_derivative(self.nodes_z[1])

    # print(self.D)

    def plot_CF(self, line1):
        if self.plotting:
            if self.nsim % self.n_plot == 0.0:
                line1 = ANN.live_plotter(self, self.n_plot_list, self.CF_plot_list, line1)
        return line1

    def evaluate_result(self):
        hit = 0
        for i in self.x:
            v = ANN.feedforward_result_norm(i[0:-1])
            print(v)
            if v >= 0.5:
                r = 1
            else:
                r = 0
            if float(r) == float(i[-1]):
                hit += 1
        print("The success percentage is %s%%" % (hit / len(self.x) * 100))

    def constant_learning_rate(self, alpha=0.001):
        self.alpha = alpha
        return alpha

    def linear_decay_learning_rate(self, b=1.0, fit_to_n=False, alpha_min=1E-5):
        if not fit_to_n:
            return self.alpha0 / (b * (self.nsim + 1))
        if fit_to_n:
            a = (alpha_min - self.alpha0) / self.NMAX
            return a * self.nsim + self.alpha0

    def quadratic_decay_learning_rate(self, b=1.0):
        return self.alpha0 / (b * (self.nsim + 1) ** 2)

    def exponential_decay_learning_rate(self, b=-1.0, fit_to_n=False, alpha_min=1E-5):
        if not fit_to_n:
            return self.alpha0 * np.exp(self.nsim * b)
        if fit_to_n:
            beta = 1 / self.NMAX * np.log(alpha_min / self.alpha0)
            return self.alpha0 * np.exp(self.nsim * beta)

    def set_adaptive_learning_rate(self, key, **args):
        print("Learning rate type set to: %s" % key)
        self.adaptive_learning_rate = True
        self.adaptive_learning_rate_type = key
        self.adaptive_learning_rate_args = args

    def unset_adaptive_learning_rate(self):
        self.adaptive_learning_rate = False

    def print_tolerances(self):
        for tol_name in self.tol:
            if tol_name == "NMAX":
                continue
            print("\t%s: %s" % (tol_name, self.tol[tol_name]["Value"]))
        print("\tLearning rate: %s" % self.alpha)

    def live_plotter(self, x_vec, y1_data, line1, identifier='', pause_time=0.001):
        """Plots the cost function over the iterations dinamically
        :param x_vec: array of iterations
        :param y1_data: array of values of the cost function
        :param line1: plotting line object
        :param identifier:
        :param pause_time: update time of the plotting frame
        :return: updated plotting line object
        """
        plt.style.use('ggplot')
        if not line1:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            self.fig = plt.figure(figsize=(13, 6))
            self.ax = self.fig.add_subplot(111)
            # create a variable for the line so we can later update it
            line1, = self.ax.plot(x_vec, y1_data, '-o', alpha=0.8)
            # update plot label/title
            plt.ylabel('Cost function [-]')
            plt.xlabel('Simulation step [-]')
            # plt.title('Title: {}'.format(identifier))
            plt.show()
        # after the figure, axis, and line are created, we only need to update the y-data
        line1.set_data(x_vec, y1_data)
        # adjust limits if new data goes beyond bounds
        if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1] or np.min(
                x_vec) <= \
                line1.axes.get_xlim()[0] or np.max(x_vec) >= line1.axes.get_xlim()[1]:
            self.ax.set_ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
            self.ax.set_xlim([np.min(x_vec) - np.std(x_vec), np.max(x_vec) + np.std(x_vec)])
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)
        # return line so we can update it again in the next iteration
        return line1

    def live_plotter_contour(self, contour, n, pause_time=0.001, xrange=None, yrange=None):
        """Plots a dynamic contour plot of the predictions (only for 2D input sets)
        Plots a dynamic contour plot of the predictions (only for 2D input sets)
        :param contour: contour object
        :param n: dimensions of the contour (n_x, n_y)
        :param pause_time: update time of the plot
        :param xrange: range of the horizontal variable
        :param yrange: range of the vertical variable
        :return:
        """
        if xrange is None:
            xrange = [-1.0, 1.0]
        if yrange is None:
            yrange = [-1.0, 1.0]
        plt.style.use('ggplot')
        nx = n[0]
        ny = n[1]
        xlist = np.linspace(xrange[0], xrange[1], nx)
        ylist = np.linspace(yrange[0], yrange[1], ny)
        plot_array = []
        for nj, j in enumerate(ylist):
            for ni, i in enumerate(xlist):
                plot_array.append(np.array([i, j]))
        for ix, i in enumerate(xlist):
            xlist[ix] = (xlist[ix] - self.xmean[0]) / self.xstd[0]
        for iy, j in enumerate(ylist):
            ylist[iy] = (xlist[iy] - self.xmean[1]) / self.xstd[1]
        nn_result = []
        for i in plot_array:
            a = float(ANN.forward_propagate(self, i)[0])
            nn_result.append(a)
        nn_result = np.array(nn_result).reshape([nx, ny])
        if not contour:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            self.fig_contour = plt.figure(figsize=(6, 6))
            self.ax_contour = self.fig_contour.add_subplot(111)
            # create a variable for the line so we can later update it
            contour = self.ax_contour.contourf(xlist, ylist, nn_result)
            # update plot label/title
            plt.ylabel('X[0] [-]')
            plt.xlabel('X[1] [-]')
            if self.add_sample_data_to_contour:
                ANN.add_sample_data_to_graph(self, self.ax_contour)
            # plt.title('Title: {}'.format(identifier))
            plt.show()
        contour = self.ax_contour.contourf(xlist, ylist, nn_result)
        if self.add_sample_data_to_contour:
            ANN.add_sample_data_to_graph(self, self.ax_contour)
        # adjust limits if new data goes beyond bounds
        # if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1] or np.min(x_vec) <= \
        #         line1.axes.get_xlim()[0] or np.max(x_vec) >= line1.axes.get_xlim()[1]:
        #     plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
        #     plt.xlim([np.min(x_vec) - np.std(x_vec), np.max(x_vec) + np.std(x_vec)])
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)
        # return line so we can update it again in the next iteration
        return 1

    def add_sample_data_to_graph(self, axes):
        x_plot = [[], []]
        y_plot = []
        for i in self.x:
            x_plot[0].append(i[0][0])
            x_plot[1].append(i[1][0])
        for i in self.y:
            y_plot.append(i[0][0])
        axes.scatter(x_plot[0], x_plot[1], c=y_plot[:])


def unwrap_self_f(arg, **kwarg):
    return ANN.individual_gradient(*arg, **kwarg)


class io(ANN):
    def print_intro(self):
        """Prints header of the ANN solver
        Prints header and some extra information of the ANN solver and the created multilayer perceptron
        :return:
        """
        print("********************************************")
        print("********* multilayer ANN solver *********")
        print("***************** vs %s *******************" % current_vs)
        print("********************************************\n")
        print("The following multilayer perceptron has been generated:")
        print("\t Dimensions: %s" % self.dim)
        print("\t Activation functions: %s" % self.activation_function_list)
        print("\t Cost function: %s" % self.cost_function_type)
