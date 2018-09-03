import csv
import numpy as np
import collections
import matplotlib.pyplot as plt
import itertools
from sklearn import datasets, cross_validation, metrics

def relu(z):
    return np.maximum(z, 0)

def relu_deriv(y):
    return 1. * (y > 0)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def read_data():
    x_train_data = []
    y_train_data = []
    x_test_data = []
    y_test_data = []
    x_test_path = '../Question_2_1/x_test.csv'
    y_test_path = '../Question_2_1/y_test.csv'
    x_train_path = '../Question_2_1/x_train.csv'
    y_train_path = '../Question_2_1/y_train.csv'

    with open(x_train_path) as x_train:
        read = csv.reader(x_train, delimiter='\n', quotechar='|')
        for x in read:
            element = x[0].split(',')
            array = list(map(int, element))
            x_train_data.append(array)
    x_train_data = np.asarray(x_train_data)

    with open(y_train_path) as y_train:
        read = csv.reader(y_train, delimiter='\n')
        for element in read:
            if element[0] == '0':
                y_train_data.append([1, 0, 0, 0])
            elif element[0] == '1':
                y_train_data.append([0, 1, 0, 0])
            elif element[0] == '2':
                y_train_data.append([0, 0, 1, 0])
            elif element[0] == '3':
                y_train_data.append([0, 0, 0, 1])
    y_train_data = np.asarray(y_train_data)

    with open(x_test_path) as x_test:
        read = csv.reader(x_test, delimiter='\n', quotechar='|')
        for x in read:
            element = x[0].split(',')
            arr = list(map(int, element))
            x_test_data.append(arr)
    x_test_data = np.asarray(x_test_data)

    with open(y_test_path) as y_test:
        read = csv.reader(y_test, delimiter='\n')
        for x in read:
            if x[0] == '0':
                y_test_data.append([1, 0, 0, 0])
            elif x[0] == '1':
                y_test_data.append([0, 1, 0, 0])
            elif x[0] == '2':
                y_test_data.append([0, 0, 1, 0])
            elif x[0] == '3':
                y_test_data.append([0, 0, 0, 1])
    y_test_data = np.asarray(y_test_data)

    x_validation_data, x_test_data, y_validation_data, y_test_data = cross_validation.train_test_split(x_test_data,
                                                                                                       y_test_data,
                                                                                                       test_size=0.4)
    return (x_train_data, x_test_data, x_validation_data, y_train_data, y_test_data, y_validation_data)

class Layer (object):
    def get_grad_params(self, x, grad_out):
        return[]

    def get_iter_params(self):
        return []

    def get_grad_input(self, y, grad_output=None, target=None):
        pass

    def get_out(self, x):
        pass


class LayerLinear (Layer):
    def __init__(self, vars_in, vars_out):
        self.weight = np.random.randn(vars_in, vars_out)*0.1
        self.bias = np.zeros(vars_out)

    def get_iter_params(self):
        iter_params = itertools.chain(np.nditer(self.weight, op_flags=['readwrite']),
                               np.nditer(self.bias, op_flags=['readwrite']))

        return iter_params

    def get_out(self, x):
        return x.dot(self.weight)+self.bias


    def get_grad_params(self, x, grad_out):
        grad_weight = x.T.dot(grad_out)
        grad_bias = np.sum(grad_out, axis=0)

        return [grad for grad in itertools.chain(np.nditer(grad_weight), np.nditer(grad_bias))]

    def get_grad_input(self, y, grad_output):
        return grad_output.dot(self.weight.T)


class LayerReLu(Layer):
    def get_out(self, x):
        return relu(x)

    def get_grad_input(self, y, grad_output):
        return np.multiply(relu_deriv(y), grad_output)


class LayerSoftMaxOut(Layer):
    def get_out(self, x):
        return softmax(x)

    def get_grad_input(self, y, grad_output):
        return (y - grad_output) / y.shape[0]

    def get_cost(self, y, cost):
        return - np.multiply(cost, np.log(y)).sum() / y.shape[0]


def step_forward(samples, layers):
    activations = [samples]
    x = samples
    for layer in layers:
        y = layer.get_out(x)
        activations.append(y)
        x=activations[-1]
    return activations


def step_back(activations, target, layers):
    gradient_parameters = collections.deque()
    gradient_output = None

    for layer in reversed(layers):
        last = activations.pop()

        if gradient_output is None:
            gradient_input = layer.get_grad_input(last, target)
        else:
            gradient_input = layer.get_grad_input(last, gradient_output)

        next = activations[-1]
        gradients = layer.get_grad_params(next, gradient_output)
        gradient_parameters.appendleft(gradients)
        gradient_output = gradient_input

    return list(gradient_parameters)


def check_gradient(layers):
    data_subset = 10
    temp_x = x_train_data[0:data_subset,:]
    temp_y = y_train_data[0:data_subset,:]

    activate = step_forward(temp_x, layers)
    gradient_parameters = step_back(activate, temp_y, layers)

    change = 0.0001

    for nix in range(len(layers)):
        layer = layers[nix]
        layer_backprop = gradient_parameters[nix]

        for e_nix, parameter in enumerate(layer.get_iter_params()):
            backprop = layer_backprop[e_nix]

            parameter += change
            cost_increase = layers[-1].get_cost(step_forward(temp_x, layers)[-1], temp_y)

            parameter -= 2*change
            cost_decrease = layers[-1].get_cost(step_forward(temp_x, layers)[-1], temp_y)

            parameter += change

            gradient_number = (cost_increase - cost_decrease)/(2*change)

            if not np.isclose(gradient_number, backprop):
                raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.
                                 format(float(gradient_number), float(backprop)))
    print('No gradient errors')

def parameter_update(layers, gradients, rate):
    for layer, layer_backprop in zip(layers, gradients):
        for parameter, gradient in zip(layer.get_iter_params(), layer_backprop):
            parameter-=rate*gradient

def costs_plot(batch_cost, train_cost, validate_cost, iterations, batches):
    batch_x = np.linspace(0, iterations, num=iterations*batches)
    iteration_x = np.linspace(1, iterations, num=iterations)

    plt.plot(batch_x, batch_cost, 'k-', linewidth=0.5, label='minibatches cost')
    plt.plot(iteration_x, train_cost, 'r-', linewidth=2, label='Training Set Cost')
    plt.plot(iteration_x, validate_cost, 'b-', linewidth=3, label='cost validation')

    plt.xlabel('iteration')
    plt.ylabel('%\\xi$', fontsize=14)
    plt.title('Cost over backprop iterations')
    a1, a2, b1, b2 = plt.axis()
    plt.axis((0, iterations, 0, 2.5))
    plt.grid()
    plt.show()



def accuracy_plot(train, validation, iterations):
    iteration_x = np.linspace(1, iterations, num=iterations)

    plt.plot(iteration_x, train, 'r-', linewidth=2, label='accuracy of set')
    plt.plot(iteration_x, validation, 'b-', linewidth=3, label='Validation accuracy')

    plt.xlabel('Iterations')
    plt.ylabel('accuracy')
    plt.title('Accuracy over backprop iteration')
    plt.legend(loc=4)
    a1, a2, b1, b2 = plt.axis()
    plt.axis((0, iterations, 0, 1.0))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    x_train_data, x_test_data, x_validation, y_train_data, y_test_data, y_validation = read_data()

    hid_neur_1 = 100
    hid_neur_2 = 40

    layers = []
    layers.append(LayerLinear(x_train_data.shape[1], hid_neur_1))
    layers.append(LayerReLu())

    layers.append(LayerLinear(hid_neur_1, hid_neur_2))
    layers.append(LayerReLu())

    layers.append(LayerLinear(hid_neur_2, y_train_data.shape[1]))
    layers.append(LayerSoftMaxOut())

    check_gradient(layers)

    size_batch = 25
    batch_number = x_train_data.shape[0]/size_batch

    batches_xy = zip(np.array_split(x_train_data, batch_number, axis=0), np.array_split(y_train_data,
                                                                                        batch_number, axis=0))
    batch_costs=[]
    train_costs=[]
    validate_costs=[]
    train_accuracy=[]
    validation_accuracy=[]

    max_iterations=100
    rate_learning = 0.01

    true_y = np.argmax(y_test_data, axis=1)
    true_train_x = np.argmax(y_train_data, axis=1)
    true_x_value = np.argmax(y_validation, axis=1)

    for iterate in range(max_iterations):
        for a, b in batches_xy:
            activations = step_forward(a, layers)
            batch_cost = layers[-1].get_cost(activations[-1], b)
            batch_costs.append(batch_cost)
            grad_params = step_back(activations, b, layers)
            parameter_update(layers, grad_params,rate_learning)

        activations = step_forward(x_train_data,layers)
        train_cost = layers[-1].get_cost(activations[-1], y_train_data)
        train_costs.append(train_cost)
        pred_y = np.argmax(activations[-1], axis=1)
        train_acc = metrics.accuracy_score(true_train_x, pred_y)
        train_accuracy.append(train_acc)

        activations = step_forward(x_validation, layers)
        validate_cost = layers[-1].get_cost(activations[-1], y_validation)
        validate_costs.append(validate_cost)
        pred_y = np.argmax(activations[-1], axis=1)
        validation_acc = metrics.accuracy_score(true_x_value, pred_y)
        validation_accuracy.append(validation_acc)

        print('iter {}: train loss {:.4f} acc {:.4f}, val loss {:.4f} acc {:.4f}'.format(iterate+1, train_cost,
                                                                                         train_acc,
                                                                                         validate_cost,
                                                                                         validation_acc))
        if len(validate_costs) > 3:
            if validate_costs[-1] >= validate_costs[-2] >= validate_costs[-3]:
                break

    iterations = iterate+1

    costs_plot(batch_costs, train_costs, validate_costs, iterations, batch_number)
    accuracy_plot(train_accuracy, validation_accuracy, iterations)

    activations = step_forward(x_test_data, layers)
    pred_y = np.argmax(activations[-1], axis=1)
    test_accuracy = metrics.accuracy_score(true_y, pred_y)
    print('Final accuracy on test: {:.4f}'.format(test_accuracy))



