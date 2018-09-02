import csv
from itertools import islice
import numpy as np


def read_data(file_name):
    data_set = []
    with open(file_name, newline='') as csvfile:
        data_file = csv.reader(csvfile)
        for row in data_file:
            data_set.append(list(float(x) for x in islice(list(row), 1, None)))
    return np.array(data_set)


def get_weights(w, b):
    eq_weight = []
#    for row in range(0, 5):
#        eq_weight_row = []
#        for column in range(0, 5):
#            eq_weight_row.append(weights[row][column] * weights[row + 5][column] * weights[row + 10][column])
#        eq_weight.append(eq_weight_row)
    eq_weight = w[0][0]*(w[5][0]*(w[10][0]+b[2][0]))+(b[1][0]*(w[10][0]+b[2][0]))
    print(eq_weight)


def get_bias(b, w):
    eq_bias = []
    #for column in range(0, 5):
    #    eq_bias.append(bias[0][column] * weights[5][column] * weights[10][column] + bias[1][column] * weights[10][column] + bias[2][column])
    eq_bias = ((b[0][0] * w[11][0])*(w[6][0]+b[1][0]))+b[2][0]
    print(eq_bias)


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

get_weights(a_weights, a_bias)
get_bias(a_bias, a_weights)
