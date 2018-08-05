# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io
from opt_utils import  *
from testCases import *

def updata_parameter(para, grads, learning_rate = 0.01):
    layer = len(para) // 2
    for i in range(1, layer):
        para["W" + str(i)] -= learning_rate * grads["dW" + str(i)]
        para["b" + str(i)] -= learning_rate * grads["db" + str(i)]
    return para

def updata_parameter_sgd(X, Y, para):
    num_iter = 1000
    m = X.shape[1]
    costs = 0
    for i in range(num_iter):
        for j in range(m):
            # forward_compute
            a, cache = forward_propagation(X[:, j].reshape(X.shape[0], 1), para)
            # cost compute
            costs = costs + compute_cost(a, Y[0, j].reshape(Y.shape[0], 1))
            # backpropagate
            grads = backward_propagation(X[:, j].reshape(X.shape[0], 1), Y[0, j].reshape(Y.shape[0], 1), cache)
            # updata
            para = updata_parameter(para, grads)
        costs /= m
        print(i, "Times, ", "cost is ", costs)
        costs = 0
    return para

def model(X, Y, para):
    num_iter = 2000
    m = X.shape[1]

    for i in range(num_iter):
        # forward_compute
        a, cache = forward_propagation(X, para)
        # cost compute
        cost = compute_cost(a, Y)
        # backpropagate
        grads = backward_propagation(X, Y, cache)
        # updata
        para = updata_parameter(para, grads)
        if i % 100 == 0:
            print(i, "Times, ", "cost is ", cost)
    return para

def main():
    plt.rcParams["figure.figsize"] = (7.0, 4.0)
    plt.rcParams["image.interpolation"] = 'nearest'
    plt.rcParams["image.cmap"] = "gray"
    # (nx, m)2,211   (1, m)
    # test has 200 examples
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    layer_dim = (train_X.shape[0], 7, 5, len(train_Y))
    para = initialize_parameters(layer_dim)
    # para = updata_parameter_sgd(train_X, train_Y, para)
    para = model(train_X, train_Y, para)
    predict(train_X, train_Y, para)
    # plt.show()

if __name__ == '__main__':
    main()