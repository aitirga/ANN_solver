'''
Created on 13 nov. 2017

@author: aitorlm
'''
import numpy as np
from scipy.optimize import minimize
import random as rd
import time
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

# This is the input function to which the GD algorithm will be applied
# Parameters

# This is my zero value

eps = 1.0E-70


class ANN:
    # This is the Neural Network class builder. 
    # Input parameters are:
    # 1. dim vector, containing the inner structure of the NN layers

    def __init__(self, dim, activation_function_list=[], eps_ini=0.12, cost_function="MSE", verbose=True):
        self.dim = dim
        self.activation_function_list = activation_function_list
        if self.activation_function_list == []:
            self.activation_function_list.append("linear")  # "l" stands for linear activation
            for i in range(1, len(dim)):
                self.activation_function_list.append("sigmoid")  # "s" stands for sigmoid activation

        self.eps_ini = eps_ini
        self.eps = 1E-200
        self.normalized = False # Boolean that tracks if input data should be normalized
        self.cost_function_type = cost_function
        self.input_set_type = "full" # By default, all the elements in X are considered
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
        self.dict_activation_function = {"linear": ANN.linear,
                                         "sigmoid": ANN.sigmoid_matrix,
                                         "ReLU": ANN.relu}
        self.dict_activation_function_derivative = {"linear": ANN.linear_derivative,
                                                    "sigmoid": ANN.sigmoid_matrix_derivative,
                                                    "ReLU": ANN.relu_derivative}
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
        ANN.build_matrices(self)
        ANN.randomize_weights(self)
        ANN.print_intro(self)
        # ANN.handle_exceptions(self)

    def print_intro(self):
        print("********************************************")
        print("********* multilayer ANN core unit *********")
        print("***************** vs 0.7 *******************")
        print("********************************************\n")
        print("The following multilayer perceptron has been generated:")
        print("\t Dimensions: %s" % self.dim)
        print("\t Activation functions: %s" % self.activation_function_list)
        print("\t Cost function: %s" % self.cost_function_type)

    # def handle_exceptions(self):
    #     if

    def use_multiprocessing(self, processes=8):
        self.multiprocessing = True
        self.n_processes = processes
        # from pathos.multiprocessing import ProcessingPool
        import pathos.multiprocessing as mp
        from joblib import Parallel, delayed

    def read_pd_array(self, v):
        full_v = []
        for i in range(v.shape[0]):
            temp_v = np.array(v.iloc[i, :].values)
            temp_v = ANN.convert_vector(self, temp_v)
            full_v.append(temp_v)
        return full_v

    def create_data_vector_x(self, x):
        print("Setting up input vectors of the training set")
        if str(type(x)) == "<class 'pandas.core.frame.DataFrame'>":
            x_temp = ANN.read_pd_array(self, x)
            self.x = x_temp
            return 1
        x_temp = []
        for i in x:
            x_temp.append(ANN.convert_vector(self, i))
        self.x = x_temp
        print("\tInput array set up, number of elements: %s" % len(self.x))
        return 1

    def add_data_element_X(self, X):
        """
        Adds an input data element to input vector X
        :param X:
        :return:
        """
        temp_numpy = np.zeros(shape=(len(X), 1))
        for i in range(0, len(X)):
            temp_numpy[i, 0] = X[i]
        self.x.append(temp_numpy)

    def create_data_vector_y(self, y):
        print("Setting up output vectors of the training set")
        if str(type(y)) == "<class 'pandas.core.frame.DataFrame'>":
            x_temp = ANN.read_pd_array(self, y)
            self.x = x_temp
            return 1
        y_temp = []
        for i in y:
            y_temp.append(ANN.convert_vector(self, i))
        self.y = y_temp
        print("Output array set up, number of elements: %s" % len(self.x))

    def add_data_element_Y(self, Y):
        temp_numpy = np.zeros(shape=(len(Y), 1))
        for i in range(0, len(Y)):
            temp_numpy[i, 0] = Y[i]
        self.y.append(temp_numpy)

    def add_sample_data(self):
        self.add_sample_data_to_contour = True

    # def convert_input_vector(self, x):
    #     """
    #     This function takes the input vector x and transforms it into the format used inside the class
    #     :param x: input vector
    #     :return: vector in class format
    #     """
    #     try:
    #         if (x.shape[0] == self.dim[0]) and len(x.shape) == 2:
    #             return x
    #         elif x.shape[0] == self.dim[0] and len(x.shape) == 1:
    #             input_vec = np.ones(shape=[self.dim[0], 1])
    #             for idx, item in enumerate(x):
    #                 input_vec[idx, 0] = item
    #             return input_vec

    def convert_vector(self, vec):
        """
        This function takes the input vector x and transforms it into the format used inside the class
        :param vec: input vector
        :return: vector in class format
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

    def normalize_v(self, v):
        temp_v = np.zeros_like(self.x[0])
        for i in range(0, self.x[0].shape[0]):
            temp_v[i, 0] = 1 / (self.xmax[i] - self.xmin[i]) * (np.float32(v[i]) - self.xmin[i])
        return temp_v

    def normalize_v_stdmean(self, v):
        temp_v = np.zeros_like(self.x[0])
        for i in range(0, self.x[0].shape[0]):
            if self.xstd[i] == 0.0:
                continue
            temp_v[i, 0] = (np.float32(v[i]) - self.xmean[i]) / self.xstd[i]
        return temp_v

    def normalize_X(self):
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
        """
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
        ''' Aitor - 11/12/18
        This function divides the sample dataset into three different datasets in order to use cross-validation and test verification
        
        Arguments:
        r_train -> Ratio of the sample space to be used as the training dataset       
        r_cs -> Ratio of the sample space to be used as the cross validation dataset
        r_test -> Ratio of the sample space to be used as the test dataset
        
        '''
        self.xtrain = []
        self.xcv = []
        self.xtest = []
        N_train = np.floor(r_train * len(self.x))
        N_cv = np.floor(r_cv * len(self.x))
        N_test = np.floor(r_test * len(self.x))
        whole_list = range(0, len(self.x))

        list_train = rd.choice(whole_list, N_train)
        list_cv = rd.choice(whole_list, N_cv)
        list_test = rd.choice(whole_list, N_test)
        print(list_train)

    def build_matrices(self):  # Build all necessary matrices that will need to be used
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

    def randomize_matrix(self, A):
        shape_m = A.shape
        randM = np.zeros(shape=shape_m)
        for i in range(0, shape_m[0]):
            for j in range(0, shape_m[1]):
                val = rd.uniform(-self.eps_ini, self.eps_ini)
                randM[i, j] = val
        # print("randomized: %s" % randM)
        # randA = np.random.rand(shape[0], shape[1])
        return randM

    def randomize_weights(self):
        id = 0
        for i in self.Theta:
            i = ANN.randomize_matrix(self, i)
            self.Theta[id] = i
            id += 1

    def print_nodes(self, X, nodes):
        print("*****************************")
        print("Nodes information:")
        print(X)
        for i in range(1, len(nodes)):
            print(nodes[i])
        print("")

    @staticmethod
    def populate_with_bias(input, A):
        for i in range(1, A.shape[0]):
            A[i] = input[i - 1, 0]
        return A

    @staticmethod
    def populate_nodes_a_from_z(z, a):
        for i in range(0, len(z)):
            for j in range(1, z[i].shape[0]):
                a[i][j - 1, 0] = z[i][j, 0]
        return a

    @staticmethod
    def populate_vector_bias_to_nobias(vec):
        a = np.ones(shape=(vec.shape[0] - 1, 1))
        for j in range(1, a.shape[0] + 1):
            a[j - 1, 0] = vec[j, 0]
        return a

    @staticmethod
    def populate_vector_nobias_to_bias(vec):
        a = np.ones(shape=(vec.shape[0] + 1, 1))
        for j in range(1, a.shape[0]):
            a[j, 0] = vec[j - 1, 0]
        return a

    @staticmethod
    def populate_nodes_z_from_a(a, z):
        for i in range(0, len(a)):
            for j in range(0, a[i].shape[0]):
                z[i][j + 1, 0] = a[i][j, 0]
        return z

    def initialize_tensor(self, A):
        for k in range(0, len(A)):
            for i in range(0, self.dim[k + 1]):
                for j in range(0, self.dim[k] + 1):
                    A[k][i, j] = 0.0
        return A

    @staticmethod
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

    @staticmethod
    def sigmoid_matrix_derivative(vec):
        """
        Calculates the derivative of the sigmoid of a given vector.
        :param vec: input vector
        :param id:
        :return:
        """
        sigmoid_vector = ANN.sigmoid_matrix(vec)
        temp_array = np.multiply(sigmoid_vector, 1 - sigmoid_vector)
        return temp_array

    @staticmethod
    def linear(A):
        return A

    @staticmethod
    def linear_derivative(vec):
        temp_array = np.ones(shape=(vec.shape[0], 1))
        return temp_array

    @staticmethod
    def relu(vec, r=0.0):
        temp_array = np.maximum(r, vec)
        return temp_array

    @staticmethod
    def relu_derivative(vec, r=0.0):
        temp_array = np.maximum(r, vec)
        temp_array[temp_array > r] = 1.0
        return temp_array

    def apply_activation_function(self, vec, idx):
        """
        Function that applies an activation function defined at "dict_activation_function" dictionary to the vector vec.
        :param vec: target of the activation function
        :param idx: index that targets the type of activation function used in layer l
        :return:
        """
        function_key = self.activation_function_list[idx]
        return self.dict_activation_function[function_key](vec)

    def apply_activation_function_derivative(self, vec, idx):
        """
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
        return self.dict_input_set[self.input_set_type](self)

    def full_input_set(self):
        set_x = range(len(self.x))
        return set_x

    def stochastic_input_set(self):
        set_x = rd.choice(range(len(self.x)))
        return [set_x]

    def batch_input_set(self):
        set_x = rd.sample(range(0, len(self.x)), self.n_batch)
        return set_x


    def forward_propagation(self, X):
        # This function performs the forward propagation algorithm using the actual weights and an input vector X
        # Populate the nodes_z matrix with the input set vector X, taking into account not to overwrite the bias term
        # Apply the first activation function
        self.nodes_z[0] = ANN.populate_vector_nobias_to_bias(X)
        temp_vec = ANN.apply_activation_function(self, self.nodes_z[0], 0)
        self.nodes_a[0] = ANN.populate_vector_bias_to_nobias(temp_vec)
        for i in range(0, self.Ndim - 1):  # We have one less weight matrix
            temp_vec = np.dot(self.Theta[i], ANN.populate_vector_nobias_to_bias(self.nodes_a[i]))
            self.nodes_z[i + 1] = ANN.populate_vector_nobias_to_bias(temp_vec)
            temp_vec = ANN.apply_activation_function(self, self.nodes_z[i + 1], i + 1)
            self.nodes_a[i + 1] = ANN.populate_vector_bias_to_nobias(temp_vec)
        return self.nodes_a

    def forward_propagate(self, x):
        if str(type(x)) == "<class 'pandas.core.frame.DataFrame'>":
            x_temp = ANN.read_pd_array(self, x)
        else:
            x_temp = ANN.convert_vector(self, x)
        if self.normalized:
            x_temp = ANN.normalize_v_stdmean(self, x_temp)
        forward_propagate = ANN.forward_propagation(self, x_temp)
        return forward_propagate[-1]

    def forward_propagation_weight(self, X, Theta):
        # This function performs the forward propagation algorithm using a given weight matrix and an input vector X
        # Populate the nodes_z matrix with the input set vector X, taking into account not to overwrite the bias term
        # Apply the first activation function
        self.nodes_z[0] = ANN.populate_vector_nobias_to_bias(X)
        temp_vec = ANN.apply_activation_function(self, self.nodes_z[0], 0)
        self.nodes_a[0] = ANN.populate_vector_bias_to_nobias(temp_vec)
        for i in range(0, self.Ndim - 1):  # We have one less weight matrix
            temp_vec = np.dot(Theta[i], ANN.populate_vector_nobias_to_bias(self.nodes_a[i]))
            self.nodes_z[i + 1] = ANN.populate_vector_nobias_to_bias(temp_vec)
            temp_vec = ANN.apply_activation_function(self, self.nodes_z[i + 1], i + 1)
            self.nodes_a[i + 1] = ANN.populate_vector_bias_to_nobias(temp_vec)
        return 1

    def mse_cost_function(self):
        ff_result = self.nodes_a[-1]
        for i in range(0, ff_result.shape[0]):
            self.val_cf += 1 / 2 * (self.y_sample_cf[i] - self.nodes_a[-1][i]) ** 2  # Min square
        return self.val_cf

    def mse_cost_function_derivative(self):
        return self.nodes_a[-1] - self.y_test

    def classification_cost_function(self):
        ff_result = self.nodes_a[-1]
        for i in range(0, ff_result.shape[0]):
            self.val_cf += -(self.y_sample_cf[i] * np.log(ff_result[i] + self.eps) + (1 - self.y_sample_cf[i]) * np.log(1 - ff_result[i] + + self.eps)) # Classification
        return self.val_cf

    def classification_cost_function_derivative(self):
        temp_array = self.nodes_a[-1] - self.y_test
        temp_array2 = np.multiply(self.nodes_a[-1], 1 - self.nodes_a[-1]) + self.eps
        #if temp_array2 < self.eps:
        #    temp_array2 = self.eps
        temp_array2 = 1 / temp_array2
        return np.multiply(temp_array, temp_array2)

    def cost_function_value(self):
        return self.dict_cost_function[self.cost_function_type](self)

    def cost_function(self):
        # Iteration over the training set
        input_set = ANN.get_input_set(self)
        N_X = len(input_set)
        self.val_cf = 0.0
        for i in input_set:
            self.x_sample_cf = self.x[i]
            self.y_sample_cf = self.y[i]
            # Run first the forward_propagation algorithm to calculate the a values
            ANN.forward_propagation(self, self.x_sample_cf)
            self.val_cf = ANN.cost_function_value(self)  # Computes the cost function
        self.val_cf = self.val_cf / N_X
        # Compute the regularization term
        reg = 0.0
        for i in range(0, self.Ndim - 1):
            for j in range(1, self.dim[i] + 1):
                for k in range(0, self.dim[i + 1]):
                    reg += self.Theta[i][k, j] ** 2
        reg *= self.lambda0 / (2 * N_X)
        cost = float(self.val_cf + reg)
        self.val_cf = cost
        return cost

    def cost_function_weight(self, X, Y, Theta):
        # Iteration over the training set
        N_X = len(X)
        self.val_cf = 0.0
        for i in range(len(X)):
            self.x_sample_cf = X[i]
            self.y_sample_cf = Y[i]
            # Run first the forward_propagation algorithm to calculate the a values
            ANN.forward_propagation_weight(self, self.x_sample_cf, Theta)
            ANN.cost_function_value(self)  # Computes the cost function
        self.val_cf = self.val_cf / N_X
        # Compute the regularization term
        reg = 0.0
        for i in range(0, self.Ndim - 1):
            for j in range(1, self.dim[i] + 1):
                for k in range(0, self.dim[i + 1]):
                    reg += Theta[i][k, j] ** 2
        reg *= self.lambda0 / (2 * N_X)
        cost = float(self.val_cf + reg)
        self.val_cf = cost
        return self.val_cf

    def write_weights(self, filename):
        # Writes the weights of the Neural Network on a text file "filename.dat" in the numpy format
        np.save(filename, self.Theta)
        print("The weight matrix has been properly saved in %s" % filename)

    def load_weights(self, filename):
        # Writes the weights of the Neural Network on a text file "filename.dat" in the numpy format
        v = np.load(filename)
        self.Theta = v
        print("The weight matrix has been properly load from %s" % filename)

    def individual_gradient(self, r):
        self.x_test = self.x[r]
        self.y_test = self.y[r]
        ANN.forward_propagation(self, self.x_test)
        self.nodes_delta[-1] = self.dict_cost_function_derivative[self.cost_function_type](
            self)  # For sigmoid activation in first layer
        try_array = ANN.apply_activation_function_derivative(self,
                                                             ANN.populate_vector_bias_to_nobias(self.nodes_z[-1]),
                                                             -1)
        try_array = np.multiply(self.nodes_delta[-1], try_array)
        self.nodes_delta[-1] = try_array

        # Backpropagate the errors
        for i in range(len(self.dim) - 1, 1, -1):
            vec = np.dot(self.Theta[i - 1].transpose(), self.nodes_delta[i])
            activation_function_derivative = ANN.apply_activation_function_derivative(self, self.nodes_z[i - 1], i - 1)
            vec = np.multiply(vec, activation_function_derivative)
            self.nodes_delta[i - 1] = ANN.populate_vector_bias_to_nobias(vec)

        # Compute the accumulative Delta
        for k in range(0, len(self.dim) - 1):
            temp_tensor = np.outer(self.nodes_delta[k + 1],
                                   ANN.populate_vector_nobias_to_bias(self.nodes_a[k]).transpose())
            self.Delta[k] += temp_tensor

    def compute_gradient(self):
        # Computes the partial derivatives of the cost function in respect to the weigths for a given training sample (Xi, Yi)
        # First we need to run the forward propagation algorithm
        # Reinitialize Delta function (?)
        # Compute the last error function (i.e. the difference between the training set Y and the computed a)
        self.D = ANN.initialize_tensor(self, self.D)
        self.Delta = ANN.initialize_tensor(self, self.Delta)
        input_set = ANN.get_input_set(self)
        N_X = len(input_set)
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
                        self.D[k][i, j] = 1 / N_X * (self.Delta[k][i, j] + self.lambda0 * self.Theta[k][i, j])
                    if j == 0:
                        self.D[k][i, j] = 1 / N_X * (self.Delta[k][i, j])
        return self.D

    def check_gradient(self, eps=1E-5):
        # print("ratio: %s" % self.Dcheck)
        ANN.compute_gradient(self)
        N_X = len(self.x)
        ThetaPlus = []
        ThetaMinus = []
        for i in range(0, len(self.dim) - 1):
            ThetaPlus.append(np.zeros(shape=(self.dim[i + 1], self.dim[i] + 1)))  # Accounted for the bias term
            ThetaMinus.append(np.zeros(shape=(self.dim[i + 1], self.dim[i] + 1)))  # Accounted for the bias term
        for k in range(0, len(self.dim) - 1):
            for j in range(0, self.dim[k] + 1):
                for i in range(0, self.dim[k + 1]):
                    ThetaPlus[k][i, j] = self.Theta[k][i, j]
                    ThetaMinus[k][i, j] = self.Theta[k][i, j]
        # print("thetaPlus: %s" %ThetaPlus)

        for k in range(0, len(self.dim) - 1):
            for j in range(0, self.dim[k] + 1):
                for i in range(0, self.dim[k + 1]):
                    # ThetaMinus[k][i,j] -= eps
                    ThetaMinus[k][i, j] -= eps
                    ThetaPlus[k][i, j] += eps
                    # print("Theta_plus %s" % ThetaPlus)
                    # print("Theta_minus %s" % ThetaMinus)
                    ThetaMinus_cost = ANN.cost_function_weight(self, self.x, self.y, ThetaMinus)
                    ThetaPlus_cost = ANN.cost_function_weight(self, self.x, self.y, ThetaPlus)
                    # ThetaMinus[k][i,j] += eps
                    ThetaPlus[k][i, j] -= eps
                    ThetaMinus[k][i, j] += eps
                    # print("Checking costs: %s %s" % (ThetaMinus_cost, ThetaPlus_cost))
                    # print("Derivative: %s" % ((ThetaPlus_cost - ThetaMinus_cost)/(2*eps)))
                    self.Dcheck[k][i, j] = (ThetaPlus_cost - ThetaMinus_cost) / (2 * eps)
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
        for i in self.x:
            v = ANN.feedforward_result_norm(i[0:-1])
            print(v)
            if v >= 0.5:
                r = 1
            else:
                r = 0
            if float(r) == float(i[-1]):
                hit += 1
        print("The success percentage is %s%%" % (hit / len(X) * 100))

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

    def constant_learning_rate(self, alpha=0.001):
        self.alpha = alpha
        return alpha

    def linear_decay_learning_rate(self, b=1.0, fit_to_n=False, alpha_min=1E-5):
        if not fit_to_n:
            return self.alpha0/(b * (self.nsim + 1))
        if fit_to_n:
            a = (alpha_min - self.alpha0)/self.NMAX
            return a * self.nsim + self.alpha0

    def quadratic_decay_learning_rate(self, b=1.0):
        return self.alpha0/(b * (self.nsim + 1)**2)

    def exponential_decay_learning_rate(self, b=-1.0, fit_to_n=False, alpha_min=1E-5):
        if not fit_to_n:
            return self.alpha0*np.exp(self.nsim*b)
        if fit_to_n:
            beta = 1/self.NMAX*np.log(alpha_min/self.alpha0)
            return self.alpha0*np.exp(self.nsim*beta)

    def set_adaptive_learning_rate(self, key, **args):
        print("Learning rate type set to: %s" % key)
        self.adaptive_learning_rate = True
        self.adaptive_learning_rate_type = key
        self.adaptive_learning_rate_args = args

    def unset_adaptive_learning_rate(self):
        self.adaptive_learning_rate = False

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

    def gradient_descent(self, alpha=1e-4, ATOL=1E-3, RTOL=1E-4, AlwaysDecrease=False,
                                                    N=100, plotting=True, n_plot=False, NMAX=1E4, lambda0=0.0, input_set_type="full",
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

        if n_plot == False:
            self.n_plot = N
        else:
            self.n_plot = n_plot
        self.Time = []
        self.CF = []
        line1 = []
        # Automatic tolerance stop (ATOL)
        self.nsim = 0
        # Velocities matrix for the momentum implementation
        ANN.print_gradient_descent_intro(self)
        if Norm:  # Specifies if the input X vector should be normalized
            if self.normalized:
                pass
            else:
                ANN.normalize_x_stdmean(self)
        else:
            pass
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
            if (ANN.evaluate_tolerances(self, ATOL, RTOL, NMAX) == True):
                break
            if self.avoid_CF:
                pass
            else:
                temp_CF = ANN.cost_function(self)
                self.CF.append(temp_CF)
            ANN.evaluate_tolerances(self, ATOL, RTOL, NMAX)
            if float(self.nsim) % N == 0.0:
                if self.avoid_CF:
                    temp_CF = ANN.cost_function(self)
                    self.CF.append(temp_CF)
                print("Learning step %s, cost function value: %s" % (self.nsim, temp_CF))
                ANN.print_tolerances(self)
                if plot_contour:
                    contour = ANN.live_plotter_contour(self, contour, n_contour)
                # ANN.check_gradient(self)
            self.nsim += 1
            if plotting:
                if float(self.nsim) % N == 0:
                    self.n_plot_list.append(self.nsim)
                    self.CF_plot_list.append(temp_CF)
                    line1 = ANN.plot_CF(self, line1)
        ANN.sim_status(self)
        plt.ioff()
    def print_tolerances(self):
        for tol_name in self.tol:
            if tol_name == "NMAX":
                continue
            print("\t%s: %s" % (tol_name, self.tol[tol_name]["Value"]))
        print("\tLearning rate: %s" % self.alpha)

    def live_plotter(self, x_vec, y1_data, line1, identifier='', pause_time=0.001):
        plt.style.use('ggplot')
        if line1 == []:
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
        if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1] or np.min(x_vec) <= \
                line1.axes.get_xlim()[0] or np.max(x_vec) >= line1.axes.get_xlim()[1]:
            self.ax.set_ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
            self.ax.set_xlim([np.min(x_vec) - np.std(x_vec), np.max(x_vec) + np.std(x_vec)])
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)
        # return line so we can update it again in the next iteration
        return line1

    def live_plotter_contour(self, contour, n, pause_time=0.001, xrange=[-1.0, 1.0], yrange=[-1.0,1.0]):
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
            xlist[ix] = (xlist[ix] - self.xmean[0])/self.xstd[0]
        for iy, j in enumerate(ylist):
            ylist[iy] = (xlist[iy] - self.xmean[1])/self.xstd[1]
        NN_result = []
        for i in plot_array:
            a = float(ANN.forward_propagate(self, i)[0])
            NN_result.append(a)
        NN_result = np.array(NN_result).reshape([nx, ny])
        if contour == []:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            self.fig_contour = plt.figure(figsize=(6, 6))
            self.ax_contour = self.fig_contour.add_subplot(111)
            # create a variable for the line so we can later update it
            contour = self.ax_contour.contourf(xlist, ylist, NN_result)
            # update plot label/title
            plt.ylabel('X[0] [-]')
            plt.xlabel('X[1] [-]')
            if self.add_sample_data_to_contour:
                ANN.add_sample_data_to_graph(self, self.ax_contour)
            # plt.title('Title: {}'.format(identifier))
            plt.show()
        contour = self.ax_contour.contourf(xlist, ylist, NN_result)
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