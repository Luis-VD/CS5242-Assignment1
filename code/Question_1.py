import csv
from itertools import islice
import numpy as np

#Constants, configure here for tuning of input
network_input = [1, 2, 3, 4, 5]


def read_data(file_name):
    data_set = []
    with open(file_name, newline='') as csvfile:
        data_file = csv.reader(csvfile)
        for row in data_file:
            data_set.append(list(float(x) for x in islice(list(row), 1, None)))
    return np.array(data_set)


def first_network_iterate(weights, biases):
    layer_one = []
    layer_two = []
    layer_three = []
    for row in range(0, 5):
        z_number = 0
        for column in range(0, 5):
            z_number += network_input[column]*weights[row][column]
        layer_one.append(z_number+biases[0][row])

    for row in range(5, 10):
        z_number = 0
        for column in range(0, 5):
            z_number += layer_one[column]*weights[row][column]
        layer_two.append(z_number+biases[1][row-5])

    for row in range(10, 15):
        z_number = 0
        for column in range(0, 5):
            z_number += layer_two[column]*weights[row][column]
        layer_three.append(z_number+biases[2][row-10])

    #print(layer_three)

    return layer_three



def init_weights(weights):
    comprised_weights = []
    for row in range(0, 5):
        weight_row = []
        for column in range(0, 5):
            weight_row.append(weights[row][column]*weights[row+5][column]*weights[row+5][column])
        comprised_weights.append(weight_row)

    #print(comprised_weights)
    return comprised_weights

def init_biases (biases):
    total_bias = []
    for column in range(0, 5):
        total_bias.append((biases[0][column]+biases[1][column]+biases[2][column])/3)

    #print(total_bias)
    return total_bias


def get_new_network_output(weights, biases):
    new_output = []
    for row in range(0, 5):
        z_number = 0
        for column in range(0, 5):
            z_number += network_input[column]*weights[row][column]
        new_output.append(z_number+biases[row])

    return new_output

def get_cost (new, initial):
    cost = np.sum(np.power(np.subtract(new, initial), 2))
    return cost

def refine_weights_biases (weights, biases, initial_output, name):
    refined_weights = weights
    refined_biases = biases
    new_network_output = get_new_network_output(weights, biases)
    original_cost = get_cost(new_network_output, initial_output)
    print(original_cost)

    for row in range(0, 5):
        for column in range (0,5):
            refined_weights[row][column] += 0.001
            while True:
                new_network_output = get_new_network_output(refined_weights, refined_biases)
                new_cost = get_cost(new_network_output, initial_output)
                if new_cost <= original_cost:
                    original_cost = new_cost
                    refined_weights[row][column] += 0.001
                    #print(new_cost)
                else:
                    refined_weights[row][column] -= 0.002
                    #print('subtracting to weight')
                    break


            while True:
                new_network_output = get_new_network_output(refined_weights, refined_biases)
                new_cost = get_cost(new_network_output, initial_output)
                if new_cost <= original_cost:
                    original_cost = new_cost
                    refined_weights[row][column] -= 0.001
                else:
                    break
    np.savetxt("../"+name+"-w.csv", refined_weights, delimiter=",")

    for row in range(0, 5):
        refined_biases[row] += 0.001
        while True:
            new_network_output = get_new_network_output(refined_weights, refined_biases)
            new_cost = get_cost(new_network_output, initial_output)
            if new_cost <= original_cost:
                original_cost = new_cost
                refined_biases[row] += 0.001
            else:
                refined_biases[row] -= 0.002
                break

        while True:
            new_network_output = get_new_network_output(refined_weights, refined_biases)
            new_cost = get_cost(new_network_output, initial_output)
            if new_cost <= original_cost:
                original_cost = new_cost
                refined_biases[row] -= 0.001
            else:
                break
    print(new_cost)
    np.savetxt("../"+name+"-b.csv", refined_biases, delimiter=",")


    print(refined_weights, refined_biases)


if __name__ == '__main__':
    a_weights = read_data('../Question_1/a/a_w.csv')
    b_weights = read_data('../Question_1/b/b_w.csv')
    c_weights = read_data('../Question_1/c/c_w.csv')
    d_weights = read_data('../Question_1/d/d_w.csv')
    e_weights = read_data('../Question_1/e/e_w.csv')

    a_bias = read_data('../Question_1/a/a_b.csv')
    b_bias = read_data('../Question_1/b/b_b.csv')
    c_bias = read_data('../Question_1/c/c_b.csv')
    d_bias = read_data('../Question_1/d/d_b.csv')
    e_bias = read_data('../Question_1/e/e_b.csv')

    first_network_output = first_network_iterate(a_weights, a_bias)
    initial_weights = init_weights(a_weights)
    initial_biases = init_biases(a_bias)
    refine_weights_biases(initial_weights, initial_biases, first_network_output, 'a')

    first_network_output = first_network_iterate(b_weights, b_bias)
    initial_weights = init_weights(b_weights)
    initial_biases = init_biases(b_bias)
    refine_weights_biases(initial_weights, initial_biases, first_network_output, 'b')

    first_network_output = first_network_iterate(c_weights, c_bias)
    initial_weights = init_weights(c_weights)
    initial_biases = init_biases(c_bias)
    refine_weights_biases(initial_weights, initial_biases, first_network_output, 'c')

    first_network_output = first_network_iterate(d_weights, d_bias)
    initial_weights = init_weights(d_weights)
    initial_biases = init_biases(d_bias)
    refine_weights_biases(initial_weights, initial_biases, first_network_output, 'd')

    first_network_output = first_network_iterate(e_weights, e_bias)
    initial_weights = init_weights(e_weights)
    initial_biases = init_biases(e_bias)
    refine_weights_biases(initial_weights, initial_biases, first_network_output, 'e')

