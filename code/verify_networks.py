import numpy as np
from utilities import *

#Constants, indicate here the csv files for input
weight_14 = '../Question_2_2/c/w-14-28-4.csv'
weight_28 = '../Question_2_2/c/w-28-6-4.csv'
weight_100 = '../Question_2_2/c/w-100-40-4.csv'
bias_14 = '../Question_2_2/c/b-14-28-4.csv'
bias_28 = '../Question_2_2/c/b-28-6-4.csv'
bias_100 = '../Question_2_2/c/b-100-40-4.csv'

#Constants, indicate here the CSV files for output
derivative_weight_28 = "../dw-28-6-4.csv"
derivative_bias_28 = "../db-28-6-4.csv"
derivative_weight_100 = "../dw-100-40-4.csv"
derivative_bias_100 = "../db-100-40-4.csv"
derivative_weight_14 = "../dw-14-28-4.csv"
derivative_bias_14 = "../db-14-28-4.csv"


def data_txt():
    data_from_file = [-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]
    data_from_file = np.array(data_from_file)
    data_from_file = data_from_file.reshape(data_from_file.shape[0], -1)
    additional_data = [0, 0, 0, 1]
    additional_data = np.array(additional_data)
    additional_data = additional_data.reshape(additional_data.shape[0], -1)
    return data_from_file, additional_data


def parameters_form(weight_name, bias, dimension_layers):
    parameters = {}
    layer_length = len(dimension_layers)

    lower = 0
    for l in range(1, layer_length):
        upper = dimension_layers[l - 1] + lower
        w_list = read_parameters_from_file(weight_name, lower, upper)
        parameters['W' + str(l)] = w_list
        lower = upper

        b_list = read_parameters_from_file(bias, l - 1, l)
        parameters['b' + str(l)] = b_list

        assert (parameters['W' + str(l)].shape == (dimension_layers[l], dimension_layers[l - 1]))
        assert (parameters['b' + str(l)].shape == (dimension_layers[l], 1))

    return parameters


def compute_gradients(x, y, parameters):
    al, data_cache = forward_propagation_network(x, parameters)
    result_gradients = backward_propagation_network(al, y, data_cache, convert=True)
    # These actions are for forward and backward propagation in utilities

    return result_gradients


def read_parameters_from_file(name, lower, upper):
    list_parameters = limited_read_file(name, lower, upper)
    list_parameters = np.array(list_parameters)
    list_parameters = np.float32(list_parameters)
    list_parameters = list_parameters.reshape(list_parameters.shape[0], -1).T
    return list_parameters


def grads_csv(name, layers_dims, grads):
    object_length = len(layers_dims)
    for l in range(1, object_length):
        dw = grads['dW' + str(l)].T
        array_to_file(name[0], dw)
        db = grads['db' + str(l)].T
        array_to_file(name[1], db)


def verify_nn_1():
    layers_dims = [14, 100, 40, 4]  # network1
    x_from_txt, y_from_txt = data_txt()
    parameters = parameters_form(weight_100, bias_100, layers_dims)
    grads = compute_gradients(x_from_txt, y_from_txt, parameters)
    name = [derivative_weight_100, derivative_bias_100]
    grads_csv(name, layers_dims, grads)


def verify_nn_2():
    layers_dims = [14, 28, 28, 28, 28, 28, 28, 4]  # network2
    X, Y = data_txt()
    parameters = parameters_form(weight_28, bias_28, layers_dims)
    grads = compute_gradients(X, Y, parameters)
    name = [derivative_weight_28, derivative_bias_28]
    grads_csv(name, layers_dims, grads)


def verify_nn_3():
    layers_dims = [14]  # network3
    for i in range(28):
        layers_dims.append(14)
    layers_dims.append(4)
    X, Y = data_txt()
    parameters = parameters_form(weight_14, bias_14, layers_dims)
    grads = compute_gradients(X, Y, parameters)

    name = [derivative_weight_14, derivative_bias_14]
    grads_csv(name, layers_dims, grads)


verify_nn_1()

verify_nn_2()

verify_nn_3()
