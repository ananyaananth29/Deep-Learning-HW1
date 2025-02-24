#!/usr/bin/env python
# EXERCISE 1

import os

os.system('pip3 install matplotlib')
os.system('pip3 install numpy')
os.system('pip3 install pandas')
os.system('pip3 install python-mnist')

import urllib.request
import pandas
import numpy as np
import matplotlib.pyplot as plt
import copy
from utils import *
#test_gradient, preprocess_medical_data, load_and_preprocess_mnist

#needed to plot plots with matplotlib in OSX

#set numpy to raise exceptions when encountering numerical errors
np.seterr(all='raise')

#this function is used to convert from integer encoding of labels to one hot encoding
# labels is an 1-D array with the integer labels from 0 to n_labels. 
def one_hot(labels, n_labels):
    return np.squeeze(np.eye(n_labels)[labels.reshape(-1)])

#Does the transpose of the last two axes of a tensor
def T(input_tensor):
    return np.swapaxes(input_tensor, -1, -2)

# ---------------- exercise 1.1 ----------------
def third_degree_polynomial(x, coeffs):
    ##your code starts here

    ##your code ends here


#test your function to make sure it is doing what is expected
validate_ex11(third_degree_polynomial)

# ** END exercise 1.1 **

#coefficients of the third degree polynomial used to generate the data
coeffs_ex1 = np.array([[3.2,-1,5,-2.2]]).T

#Training data input (you are going to use for exercises 1 and 2)
x_ex1_train = np.expand_dims(np.arange(0,2,0.1), axis = 1)
x_ex1_val = np.expand_dims(np.arange(0,2,0.1), axis = 1)


#Target data
y_ex1_train = third_degree_polynomial(x_ex1_train, coeffs_ex1)
y_ex1_val = third_degree_polynomial(x_ex1_val,coeffs_ex1) 

#add noise to target data
np.random.seed(1)
epsilon_ex1 = 0.35
y_ex1_train = y_ex1_train + epsilon_ex1 * np.random.normal(size = x_ex1_train.shape)
y_ex1_val = y_ex1_val + epsilon_ex1 * np.random.normal(size = x_ex1_val.shape)

# ** END exercise 1.1 **

def return_vars_for_later():
    return x_ex1_train, y_ex1_train

# ---------------- exercise 1.2 ----------------
def fit_func(inputs, targets, degree):
    # initializing X
    X = np.zeros([inputs.shape[0], degree+1])
    ##your code starts here

    ##your code ends here

# ** END exercise 1.2 **


#testing your function to make sure it is doing what is expected
validate_ex12(fit_func, x_ex1_train, y_ex1_train)

# ---------------- exercise 1.3 ----------------
def any_degree_polynomial(x, constants_vector):
    ##your code starts here

    ##your code ends here

# ** END exercise 1.3 **


# ---------------- exercise 1.4 ----------------
#Fit a 3rd degree polynomial to data and visualize the result of your fit
##your code starts here

##your code ends here

# ** END exercise 1.4 **


# ---------------- exercise 1.5 ----------------
def mse(predicted_values, targets):
    ##your code starts here
    
    ##your code ends here


#test your function to make sure it is doing what is expected
validate_ex15(mse, y_ex1_val[:20,:],y_ex1_train)

# ** END exercise 1.5 **

# ---------------- exercise 1.7 ----------------
##your code starts here

##your code ends here


# **** END Exercise 1 ****