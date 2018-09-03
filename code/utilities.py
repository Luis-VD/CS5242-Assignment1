import csv
import numpy as np
import matplotlib.pyplot as plt
import math

# Constants used, configure here the location of the CSV files
x_train_path = '../Question_2_1/x_train.csv'
x_test_path = '../Question_2_1/x_test.csv'
y_train_path = '../Question_2_1/y_train.csv'
y_test_path = '../Question_2_1/y_test.csv'


def load_data():  # Read the data from the above paths
    x_train = read_file(x_train_path)
    train_set_x = np.array(x_train)
    train_set_x = train_set_x.astype(float)
    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T
    y_train = read_file(y_train_path)
    train_set_y = np.array(y_train)
    train_set_y = train_set_y.astype(float)
    train_set_y = expanded_y(train_set_y)
    train_set_y = train_set_y.reshape(train_set_y.shape[0], -1).T
    x_test = read_file(x_test_path)
    test_set_x = np.array(x_test)
    test_set_x = test_set_x.astype(float)
    test_set_x = test_set_x.reshape(test_set_x.shape[0], -1).T
    y_test = read_file(y_test_path)
    test_set_y = np.array(y_test)
    test_set_y = test_set_y.astype(float)
    test_set_y = expanded_y(test_set_y)
    test_set_y = test_set_y.reshape(test_set_y.shape[0], -1).T

    return train_set_x, train_set_y, test_set_x, test_set_y


def limited_read_file(name, lower, upper):
    file_list = list()
    with open(name) as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if (i >= lower) and (i < upper):
                row.pop(0)
                file_list.append(row)
            i = i + 1
    return file_list


# csv file reading
def read_file(name):
    file_list = list()
    with open(name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            file_list.append(row)
    return file_list


# convert y to expanded form
def expanded_y(input_y):
    m = input_y.shape[0]
    converted_y = np.zeros((m, 4))
    for i in range(0, m):
        if input_y[i][0] == 0.0:
            converted_y[i] = [1, 0, 0, 0]
        elif input_y[i][0] == 1.0:
            converted_y[i] = [0, 1, 0, 0]
        elif input_y[i][0] == 2.0:
            converted_y[i] = [0, 0, 1, 0]
        elif input_y[i][0] == 3.0:
            converted_y[i] = [0, 0, 0, 1]
    return converted_y


def array_to_file(name, arr):
    with open(name, 'ab') as f:
        np.savetxt(f, arr, delimiter=",")  # To write CSV files comma delimited


def use_softmax(layer_z):
    shift_z = layer_z - np.max(layer_z)
    exponential = np.exp(shift_z)
    sum_z = np.sum(exponential, axis=0, keepdims=True)
    exponential = exponential / sum_z
    assert (exponential.shape == layer_z.shape)
    resulting = layer_z
    return exponential, resulting


def init_weight_bias(dimensions):
    parameters = {}
    length = len(dimensions)

    for l in range(1, length):
        parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l - 1]) * np.sqrt(2.0 / dimensions[l])
        parameters['b' + str(l)] = np.zeros((dimensions[l], 1))  # Usage of Xavier Initialization

        assert (parameters['W' + str(l)].shape == (dimensions[l], dimensions[l - 1]))
        assert (parameters['b' + str(l)].shape == (dimensions[l], 1))

    return parameters


def linear_forward(value, weights, bias):
    result_z = np.dot(weights, value) + bias
    assert (result_z.shape == (weights.shape[0], value.shape[1]))
    cache = (value, weights, bias)
    return result_z, cache


def relu(z_value):  # Activation function for ReLu
    a_value = np.maximum(0, z_value)
    assert (a_value.shape == z_value.shape)
    cache = z_value
    return a_value, cache


def backwards_relu(diff_a, buffer):
    temp_z = buffer
    diff_z = np.array(diff_a, copy=True)  # Parses the object to correct type
    diff_z[temp_z <= 0] = 0  # In case of 0 or lower differentials
    assert (diff_z.shape == temp_z.shape)
    return diff_z


def forward_activation(previous_value, forward_weight, bias, value_active):
    if value_active == "softmax":
        result_z, linear_cache = linear_forward(previous_value, forward_weight, bias)
        result_a, activation_cache = use_softmax(result_z)
    elif value_active == "relu":
        result_z, linear_cache = linear_forward(previous_value, forward_weight, bias)
        result_a, activation_cache = relu(result_z)

    assert (result_a.shape == (forward_weight.shape[0], previous_value.shape[1]))
    cache = (linear_cache, activation_cache)

    return result_a, cache


def forward_propagation_network(value_x, prop_params):
    caches = []
    a_value = value_x
    params_length = len(prop_params) // 2

    for l in range(1, params_length):
        prev_a = a_value
        a_value, cache = forward_activation(prev_a, prop_params['W' + str(l)],
                                            prop_params['b' + str(l)], value_active="relu")  # reLu Usage
        caches.append(cache)
    after_layer, cache = forward_activation(a_value, prop_params['W' + str(params_length)],
                                            prop_params['b' + str(params_length)],
                                            value_active="softmax")  # Softmax usage
    caches.append(cache)
    assert (after_layer.shape == (4, value_x.shape[1]))
    return after_layer, caches


def compute_cross_entropy_cost(after_layer, y_param):
    shape_y = y_param.shape[1]
    calc_loss = y_param * np.log(after_layer)
    calc_cost = -np.sum(calc_loss) / shape_y
    calc_cost = np.squeeze(calc_cost)
    assert (calc_cost.shape == ())
    return calc_cost


def linear_backward(deriv_z, cache):
    prev_a, weight, bias = cache
    m = prev_a.shape[1]

    deriv_w = np.dot(deriv_z, prev_a.T) / m
    db = np.sum(deriv_z, axis=1, keepdims=True) / m
    deriv_a_prev = np.dot(weight.T, deriv_z)
    assert (deriv_a_prev.shape == prev_a.shape)
    assert (deriv_w.shape == weight.shape)
    assert (db.shape == bias.shape)
    return deriv_a_prev, deriv_w, db


def softmax_backward(layer, val_y, buffer):
    linear_cache, activation_cache = buffer
    deriv_z = layer - val_y
    deriv_prev_a, deriv_w, deriv_bias = linear_backward(deriv_z, linear_cache)
    return deriv_prev_a, deriv_w, deriv_bias


def backward_activation_relu(deriv_a, cache):
    linear_cache, activation_cache = cache
    deriv_z = backwards_relu(deriv_a, activation_cache)
    deriv_prev_a, deriv_weight, deriv_bias = linear_backward(deriv_z, linear_cache)

    return deriv_prev_a, deriv_weight, deriv_bias


def divide_batches(train_x_value, train_y_value, test_x_value, test_y_value, batch_size=64, seed=0):
    np.random.seed(seed)
    shape_x_train = train_x_value.shape[1]
    shape_x_test = test_x_value.shape[1]
    batch_train = []
    batch_test = []

    # Step 1: Shuffle (X, Y)
    perm = list(np.random.permutation(shape_x_train))
    train_x_shuffle = train_x_value[:, perm]
    train_y_shuffle = train_y_value[:, perm].reshape((4, shape_x_train))
    perm = list(np.random.permutation(shape_x_test))
    test_x_shuffle = test_x_value[:, perm]
    test_y_shuffle = test_y_value[:, perm].reshape((4, shape_x_test))

    # Step 2: Partition (shuffled_X, shuffled_Y)
    train_batches_completed = math.floor(shape_x_train / batch_size)
    for b in range(0, train_batches_completed):
        train_x_batch = train_x_shuffle[:, (b * batch_size): ((b + 1) * batch_size)]
        train_y_batch = train_y_shuffle[:, (b * batch_size): ((b + 1) * batch_size)]
        train_batch = (train_x_batch, train_y_batch)
        batch_train.append(train_batch)

    if shape_x_train % batch_size != 0:
        train_x_batch = train_x_shuffle[:, (train_batches_completed * batch_size): (
                (train_batches_completed * batch_size) + (shape_x_train % batch_size))]
        train_y_batch = train_y_shuffle[:, (train_batches_completed * batch_size): (
                (train_batches_completed * batch_size) + (shape_x_train % batch_size))]
        train_batch = (train_x_batch, train_y_batch)
        batch_train.append(train_batch)

    test_batches_completed = math.floor(shape_x_test / batch_size)
    for b in range(0, test_batches_completed):
        test_x_batch = test_x_shuffle[:, (b * batch_size): ((b + 1) * batch_size)]
        test_y_batch = test_y_shuffle[:, (b * batch_size): ((b + 1) * batch_size)]
        test_batch = (test_x_batch, test_y_batch)
        batch_test.append(test_batch)

    if shape_x_test % batch_size != 0:
        test_x_batch = test_x_shuffle[:, (test_batches_completed * batch_size): (
                (test_batches_completed * batch_size) + (shape_x_test % batch_size))]
        test_y_batch = test_y_shuffle[:, (test_batches_completed * batch_size): (
                (test_batches_completed * batch_size) + (shape_x_test % batch_size))]
        test_batch = (test_x_batch, test_y_batch)
        batch_test.append(test_batch)

    return batch_train, batch_test


def backward_propagation_network(a_layer, val_y, layer_cache, convert=False):
    gradients = {}
    val_length = len(layer_cache)

    current_cache = layer_cache[val_length - 1]  # Activating the Nth layer of softmax
    gradients["dA" + str(val_length)], gradients["dW" + str(val_length)], \
        gradients["db" + str(val_length)] = softmax_backward(a_layer, val_y, current_cache)

    # L-1 layers Relu backward activation
    for l in reversed(range(val_length - 1)):
        current_cache = layer_cache[l]
        previ_deriv_a, previous_deriv_w, temp_bias = backward_activation_relu(gradients["dA" + str(l + 2)],
                                                                              current_cache)
        if convert:
            gradients["dA" + str(l + 1)] = np.float32(previ_deriv_a)
            gradients["dW" + str(l + 1)] = np.float32(previous_deriv_w)
            gradients["db" + str(l + 1)] = np.float32(temp_bias)
        else:
            gradients["dA" + str(l + 1)] = previ_deriv_a
            gradients["dW" + str(l + 1)] = previous_deriv_w
            gradients["db" + str(l + 1)] = temp_bias

    return gradients


def weight_bias_update(weight_bias, gradients, rate):
    parameter_length = len(weight_bias) // 2

    for l in range(parameter_length):
        weight_bias["W" + str(l + 1)] = weight_bias["W" + str(l + 1)] - rate * gradients["dW" + str(l + 1)]
        weight_bias["b" + str(l + 1)] = weight_bias["b" + str(l + 1)] - rate * gradients["db" + str(l + 1)]

    return weight_bias


def velocity_init(veloc):
    veloc_length = len(veloc) // 2
    vel = {}

    for l in range(veloc_length):
        vel["dW" + str(l + 1)] = np.zeros(veloc['W' + str(l + 1)].shape)
        vel["db" + str(l + 1)] = np.zeros(veloc['b' + str(l + 1)].shape)

    return vel


def momentum_weight_bias(values, gradients, vel, beta, learning_rate):
    values_length = len(values) // 2

    for l in range(values_length):
        vel["dW" + str(l + 1)] = (beta * vel["dW" + str(l + 1)]) + ((1 - beta) * gradients['dW' + str(l + 1)])
        vel["db" + str(l + 1)] = (beta * vel["db" + str(l + 1)]) + ((1 - beta) * gradients['db' + str(l + 1)])

        values["W" + str(l + 1)] = values["W" + str(l + 1)] - (learning_rate * vel["dW" + str(l + 1)])
        values["b" + str(l + 1)] = values["b" + str(l + 1)] - (learning_rate * vel["db" + str(l + 1)])

    return values, vel


def model_network(trained_x, trained_y, tested_x, tested_y, dimension_layers, rate=0.1, iterations=2000,
                  printing_costs=False, beta=0.9, optimizer=None, batch_size=0):
    np.random.seed(1)
    training_costs = []
    training_accuracies = []
    testing_costs = []
    test_accuracies = []
    define_seed = 10

    parameters = init_weight_bias(dimension_layers)

    if optimizer == "momentum":
        v = velocity_init(parameters)

    for i in range(0, iterations + 1):
        if batch_size > 0:
            define_seed = define_seed + 1
            training_batch, testing_batch = divide_batches(trained_x, trained_y, tested_x, tested_y, batch_size,
                                                           define_seed)
            for sub_batch in training_batch:
                (x_sub_batch, y_sub_batch) = sub_batch
                a_layer, caches = forward_propagation_network(x_sub_batch, parameters)
                train_cost = compute_cross_entropy_cost(a_layer, y_sub_batch)
                grads = backward_propagation_network(a_layer, y_sub_batch, caches)

                if optimizer == "momentum":
                    parameters, v = momentum_weight_bias(parameters, grads, v, beta, rate)
                train_accuracy = predict(x_sub_batch, y_sub_batch, parameters)

            for sub_batch in testing_batch:
                (x_sub_batch, y_sub_batch) = sub_batch
                a_layer, caches = forward_propagation_network(x_sub_batch, parameters)
                test_cost = compute_cross_entropy_cost(a_layer, y_sub_batch)
                test_accuracy = predict(x_sub_batch, y_sub_batch, parameters)

        else:
            a_layer, caches = forward_propagation_network(trained_x, parameters)
            train_cost = compute_cross_entropy_cost(a_layer, trained_y)
            grads = backward_propagation_network(a_layer, trained_y, caches)
            if optimizer == "momentum":
                parameters, v = momentum_weight_bias(parameters, grads, v, beta, rate)
            else:
                parameters = weight_bias_update(parameters, grads, rate)

            train_accuracy = predict(trained_x, trained_y, parameters)
            test_a_layer, caches_test = forward_propagation_network(tested_x, parameters)
            test_cost = compute_cross_entropy_cost(test_a_layer, tested_y)
            test_accuracy = predict(tested_x, tested_y, parameters)

        if printing_costs and i % 100 == 0:
            print("Train cost after iteration %i:" % i + str(train_cost))
            print("Train accuracy after iteration %i:" % i + str(train_accuracy))

        training_costs.append(train_cost)
        training_accuracies.append(train_accuracy)

        if printing_costs and i % 100 == 0:
            print("Test cost after iteration %i:" % i + str(test_cost))
            print("Test accuracy after iteration %i:" % i + str(test_accuracy))

        testing_costs.append(test_cost)
        test_accuracies.append(test_accuracy)

    # plot the train cost
    plt.plot(np.squeeze(training_costs), label='train data')
    plt.legend(loc='upper right')
    plt.ylabel('cost')
    plt.xlabel('iterations (per 100)')
    plt.title("Learning rate = " + str(rate))
    plt.show()

    # plot the test cost
    plt.plot(np.squeeze(testing_costs), label='test data')
    plt.legend(loc='upper right')
    plt.ylabel('cost')
    plt.xlabel('iterations (per 100)')
    plt.title("Learning rate = " + str(rate))
    plt.show()

    # plot the train and test accuracy
    plt.plot(np.squeeze(training_accuracies), '-b', label='train data')
    plt.plot(np.squeeze(test_accuracies), '-r', label='test data')
    plt.legend(loc='lower right')
    plt.ylabel('accuracy')
    plt.xlabel('iterations (per 100)')
    plt.title("Learning rate = " + str(rate))
    plt.show()

    return parameters


def predict(predict_x, predict_y, data):
    x_shape = predict_x.shape[1]
    length_data = len(data) // 2
    zeros_init = np.zeros((4, x_shape))
    chances, buffers = forward_propagation_network(predict_x, data)
    for i in range(0, chances.shape[0]):
        for j in range(0, chances.shape[1]):
            if chances[i, j] > 0.8:
                zeros_init[i, j] = 1
            else:
                zeros_init[i, j] = 0

    accuracy = np.sum((zeros_init == predict_y) / (4 * x_shape))
    return accuracy
