import numpy as np
from utilities import *

# initializations
np.random.seed(1)

def build_nn_1():
    layers_dims = [14, 100, 40, 4]  # network1
    train_x, train_y, test_x, test_y = load_data()
    parameters = network_model(train_x, train_y, test_x, test_y, layers_dims, learning_rate=0.1, num_iterations=2000, print_cost=True)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)

def build_nn_2():
    layers_dims = [14,28,28,28,28,28,28,4] #network2
    train_x, train_y, test_x, test_y = load_data()
    parameters = network_model(train_x, train_y,test_x, test_y, layers_dims, learning_rate=0.1, num_iterations=1300, print_cost=True, beta=0.9, optimizer="momentum")
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)

def build_nn_3():
    layers_dims = [14] #network3
    for i in range(28):
        layers_dims.append(14)
    layers_dims.append(4)
    print(layers_dims)
    train_x, train_y, test_x, test_y = load_data()
    parameters = network_model(train_x, train_y, test_x, test_y, layers_dims, learning_rate=0.01, num_iterations=700, print_cost=True, beta=0.9, optimizer="momentum", mini_batch_size=64)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)


build_nn_1()

#build_nn_2()

#build_nn_3()