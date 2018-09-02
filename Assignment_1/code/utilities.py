import csv
import numpy as np
import matplotlib.pyplot as plt
import math

# csv file reading
def read_file_within_limit(name,lower,upper):
    l = list();
    with open(name) as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if ((i >= lower) and (i < upper)):
                row.pop(0)
                l.append(row)
            i = i + 1
    return l

# csv file reading
def read_file(name):
    l =list();
    with open(name,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            l.append(row)
    return l

# csv file writing
def write_array_to_file(name, arr):
    with open(name, 'ab') as f:
        np.savetxt(f, arr, delimiter=",")

#convert y to expanded form
def convert_y_to_expanded_form(Y):
    m = Y.shape[0]
    converted_y = np.zeros((m,4))
    for i in range(0,m):
        if Y[i][0] == 0.0:
            converted_y[i] = [1, 0, 0, 0]
        elif Y[i][0] == 1.0:
            converted_y[i] = [0, 1, 0, 0]
        elif Y[i][0] == 2.0:
            converted_y[i] = [0, 0, 1, 0]
        elif Y[i][0] == 3.0:
            converted_y[i] = [0, 0, 0, 1]
    return converted_y

def load_data():
    #use the right path for the files
    #here it is in the current working directory
    #training set
    x_train = read_file("../Question_2_1/x_train.csv")
    train_set_x = np.array(x_train)
    train_set_x = train_set_x.astype(float)
    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T

    y_train = read_file("../Question_2_1/y_train.csv")
    train_set_y = np.array(y_train)
    train_set_y = train_set_y.astype(float)
    train_set_y = convert_y_to_expanded_form(train_set_y)
    train_set_y = train_set_y.reshape(train_set_y.shape[0], -1).T

    #test set
    x_test = read_file("../Question_2_1/x_test.csv")
    test_set_x = np.array(x_test)
    test_set_x = test_set_x.astype(float)
    test_set_x = test_set_x.reshape(test_set_x.shape[0], -1).T

    y_test = read_file("../Question_2_1/y_test.csv")
    test_set_y = np.array(y_test)
    test_set_y = test_set_y.astype(float)
    test_set_y = convert_y_to_expanded_form(test_set_y)
    test_set_y = test_set_y.reshape(test_set_y.shape[0],-1).T

    return train_set_x, train_set_y, test_set_x, test_set_y


def initialize_parameters_w_and_b(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        #xavier initialization for Relu
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2.0/layer_dims[l])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

#softmax activation function
def softmax(Z):
    shiftZ = Z - np.max(Z)
    A = np.exp(shiftZ)
    sum = np.sum(A,axis=0,keepdims=True)
    A = A / sum

    assert (A.shape == Z.shape)

    cache = Z
    return A,cache

#relu activation function
def relu(Z):
    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def forward_activation(A_prev, W, b, activation):
    if activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def forward_propagation_network(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    #relu for L-1 layers
    for l in range(1, L):
        A_prev = A
        A, cache = forward_activation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)

    # softmax for Lth layer
    AL, cache = forward_activation(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="softmax")
    caches.append(cache)

    assert (AL.shape == (4, X.shape[1]))

    return AL, caches


def compute_cross_entropy_cost(AL, Y):
    m = Y.shape[1]

    loss = Y * np.log(AL);
    cost = -np.sum(loss) / m

    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def backward_activation_softmax(AL, Y, cache):
    linear_cache, activation_cache = cache
    dZ = AL - Y
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def backward_activation_relu(dA, cache):
    linear_cache, activation_cache = cache
    dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def backward_propagation_network(AL, Y, caches, convert=False):
    grads = {}
    L = len(caches)
    m = AL.shape[1]

    # Lth layer softmax backward activation
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = backward_activation_softmax(AL, Y, current_cache)

    # L-1 layers Relu backward activation
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_activation_relu(grads["dA" + str(l + 2)], current_cache)
        if(convert == True):
            grads["dA" + str(l + 1)] = np.float32(dA_prev_temp)
            grads["dW" + str(l + 1)] = np.float32(dW_temp)
            grads["db" + str(l + 1)] = np.float32(db_temp)
        else:
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters_w_and_b(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters['W' + str(l + 1)].shape))
        v["db" + str(l + 1)] = np.zeros((parameters['b' + str(l + 1)].shape))

    return v


def update_parameters_w_and_b_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        v["dW" + str(l + 1)] = (beta * v["dW" + str(l + 1)]) + ((1 - beta) * grads['dW' + str(l + 1)])
        v["db" + str(l + 1)] = (beta * v["db" + str(l + 1)]) + ((1 - beta) * grads['db' + str(l + 1)])

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (learning_rate * v["dW" + str(l + 1)])
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (learning_rate * v["db" + str(l + 1)])

    return parameters, v


def create_mini_batches(X_train, Y_train, X_test, Y_test, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X_train.shape[1]
    n = X_test.shape[1]
    mini_batches_train = []
    mini_batches_test = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X_train = X_train[:, permutation]
    shuffled_Y_train = Y_train[:, permutation].reshape((4, m))

    permutation = list(np.random.permutation(n))
    shuffled_X_test = X_test[:, permutation]
    shuffled_Y_test = Y_test[:, permutation].reshape((4, n))

    # Step 2: Partition (shuffled_X, shuffled_Y)
    num_complete_minibatches_train = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches_train):
        mini_batch_X_train = shuffled_X_train[:, (k * mini_batch_size): ((k + 1) * mini_batch_size)]
        mini_batch_Y_train = shuffled_Y_train[:, (k * mini_batch_size): ((k + 1) * mini_batch_size)]
        mini_batch_train = (mini_batch_X_train, mini_batch_Y_train)
        mini_batches_train.append(mini_batch_train)

    if m % mini_batch_size != 0:
        mini_batch_X_train = shuffled_X_train[:,
                       (num_complete_minibatches_train * mini_batch_size): ((num_complete_minibatches_train * mini_batch_size) + (m % mini_batch_size))]
        mini_batch_Y_train = shuffled_Y_train[:,
                       (num_complete_minibatches_train * mini_batch_size): ((num_complete_minibatches_train * mini_batch_size) + (m % mini_batch_size))]
        mini_batch_train = (mini_batch_X_train, mini_batch_Y_train)
        mini_batches_train.append(mini_batch_train)

    num_complete_minibatches_test = math.floor(n / mini_batch_size)
    for k in range(0, num_complete_minibatches_test):
        mini_batch_X_test = shuffled_X_test[:, (k * mini_batch_size): ((k + 1) * mini_batch_size)]
        mini_batch_Y_test = shuffled_Y_test[:, (k * mini_batch_size): ((k + 1) * mini_batch_size)]
        mini_batch_test = (mini_batch_X_test, mini_batch_Y_test)
        mini_batches_test.append(mini_batch_test)

    if n % mini_batch_size != 0:
        mini_batch_X_test = shuffled_X_test[:,
                       (num_complete_minibatches_test * mini_batch_size): ((num_complete_minibatches_test * mini_batch_size) + (n % mini_batch_size))]
        mini_batch_Y_test = shuffled_Y_test[:,
                       (num_complete_minibatches_test * mini_batch_size): ((num_complete_minibatches_test * mini_batch_size) + (n % mini_batch_size))]
        mini_batch_test = (mini_batch_X_test, mini_batch_Y_test)
        mini_batches_test.append(mini_batch_test)

    return mini_batches_train, mini_batches_test

def network_model(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate=0.1, num_iterations=2000, print_cost=False, beta = 0.9, optimizer=None, mini_batch_size=0):
    seed = 10
    np.random.seed(1)
    train_costs = []
    train_accuracies = []
    test_costs = []
    test_accuracies = []

    parameters = initialize_parameters_w_and_b(layers_dims)

    if optimizer == "momentum":
        v = initialize_velocity(parameters)

    for i in range(0, num_iterations+1):
        if(mini_batch_size > 0):
            seed = seed + 1
            minibatches_train, minibatches_test = create_mini_batches(X_train, Y_train, X_test, Y_test, mini_batch_size, seed)
            for minibatch in minibatches_train:
                (minibatch_X, minibatch_Y) = minibatch

                AL, caches = forward_propagation_network(minibatch_X, parameters)

                train_cost = compute_cross_entropy_cost(AL, minibatch_Y)

                grads = backward_propagation_network(AL, minibatch_Y, caches)

                if optimizer == "momentum":
                    parameters, v = update_parameters_w_and_b_with_momentum(parameters, grads, v, beta, learning_rate)

                train_accuracy = predict(minibatch_X, minibatch_Y, parameters)

            for minibatch in minibatches_test:
                (minibatch_X, minibatch_Y) = minibatch

                AL, caches = forward_propagation_network(minibatch_X, parameters)

                test_cost = compute_cross_entropy_cost(AL, minibatch_Y)

                test_accuracy = predict(minibatch_X, minibatch_Y, parameters)

        else:
            AL, caches = forward_propagation_network(X_train, parameters)

            train_cost = compute_cross_entropy_cost(AL, Y_train)

            grads = backward_propagation_network(AL, Y_train, caches)

            if optimizer == "momentum":
                parameters, v = update_parameters_w_and_b_with_momentum(parameters, grads, v, beta, learning_rate)
            else:
                parameters = update_parameters_w_and_b(parameters, grads, learning_rate)

            train_accuracy = predict(X_train, Y_train, parameters)

            AL_test, caches_test = forward_propagation_network(X_test, parameters)

            test_cost = compute_cross_entropy_cost(AL_test, Y_test)

            test_accuracy = predict(X_test, Y_test, parameters)

        if print_cost and i % 100 == 0:
            print("Train cost after iteration %i:" % (i) + str(train_cost))
            print("Train accuracy after iteration %i:" % (i) + str(train_accuracy))
        if print_cost and i % 100 == 0:
            train_costs.append(train_cost)
            train_accuracies.append(train_accuracy)

        if print_cost and i % 100 == 0:
            print("Test cost after iteration %i:" % (i) + str(test_cost))
            print("Test accuracy after iteration %i:" % (i) + str(test_accuracy))
        if print_cost and i % 100 == 0:
            test_costs.append(test_cost)
            test_accuracies.append(test_accuracy)



    # plot the train cost
    plt.plot(np.squeeze(train_costs), label='train data')
    plt.legend(loc='upper right')
    plt.ylabel('cost')
    plt.xlabel('iterations (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    # plot the test cost
    plt.plot(np.squeeze(test_costs), label='test data')
    plt.legend(loc='upper right')
    plt.ylabel('cost')
    plt.xlabel('iterations (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    # plot the train and test accuracy
    plt.plot(np.squeeze(train_accuracies),'-b', label='train data')
    plt.plot(np.squeeze(test_accuracies),'-r', label='test data')
    plt.legend(loc='lower right')
    plt.ylabel('accuracy')
    plt.xlabel('iterations (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


def predict(X, Y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((4, m))

    prob, caches = forward_propagation_network(X, parameters)

    for i in range(0, prob.shape[0]):
        for j in range(0,prob.shape[1]):
            if prob[i, j] > 0.8:
                p[i, j] = 1
            else:
                p[i, j] = 0

    accuracy = np.sum((p == Y) / (4 * m))
    return accuracy