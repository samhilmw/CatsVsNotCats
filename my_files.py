import numpy as np
import h5py

# Domain Real numbers, Range open interval 0 to 1
def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

# Non linear activation function
def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A, cache

# Used in back prop
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)      
    dZ[Z <= 0] = 0
    return dZ

# Used in back prop
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

# Loading data from h5 files and storing as numpy array
def load_data():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


