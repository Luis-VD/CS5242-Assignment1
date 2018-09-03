import numpy as np
from utilities import *


def network_one():
    layers_dims = [14, 100, 40, 4]  # Dimension for the layers in Network 1
    train_x, train_y, test_x, test_y = load_data()
    parameters = model_network(train_x, train_y, test_x, test_y, layers_dims, rate=0.1, iterations=2000, printing_costs=True)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)

def network_two():
    layers_dims = [14,28,28,28,28,28,28,4] #Dimensions for the layers in Network 2
    train_x, train_y, test_x, test_y = load_data()
    parameters = model_network(train_x, train_y, test_x, test_y, layers_dims, rate=0.1, iterations=1300, printing_costs=True, beta=0.9, optimizer="momentum")
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)

def network_three():
    layers_dims = [14] #Dimensions for the layers in Network 3
    for i in range(28):
        layers_dims.append(14)
    layers_dims.append(4)
    print(layers_dims)
    train_x, train_y, test_x, test_y = load_data()
    parameters = model_network(train_x, train_y, test_x, test_y, layers_dims, rate=0.01, iterations=700, printing_costs=True, beta=0.9, optimizer="momentum", batch_size=64)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)


# initializations
np.random.seed(3)

network_one()

network_two()

network_three()