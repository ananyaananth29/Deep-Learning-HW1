#!/usr/bin/env python
# EXERCISE 2

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
from assignment1_ex1 import return_vars_for_later, mse
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


def initialize_parameters_ex2(n_inputs, n_hidden_nodes, n_outputs):
    np.random.seed(1)
    #initialize weights centered in 0
    weights_1 = np.random.normal(0,0.5,[n_inputs,n_hidden_nodes])
    # initialize bias with a small positive value to reduce amount of dead neurons
    bias_1 = np.random.normal(0.1,0,[n_hidden_nodes])
    #initialize weights centered in 0
    weights_2 = np.random.normal(0,0.5,[n_hidden_nodes,n_outputs])
    # initialize bias with a small positive value to reduce amount of dead neurons
    bias_2 = np.random.normal(0.1,0,[n_outputs])
    return {'weights_1':weights_1, 'weights_2':weights_2, 'bias_1':bias_1, 'bias_2':bias_2}


# ---------------- exercise 2.2 ----------------
def two_layer_network_forward(inputs, parameters, return_intermediary_results = False):
    
    ##your code starts here    
    W1 = parameters['weights_1']
    b1 = parameters['bias_1']
    W2 = parameters['weights_2']
    b2 = parameters['bias_2']

    out_1 = inputs @ W1 + b1
    out_1_relu = np.maximum(0, out_1)  # ReLU activation

    out_2 = out_1_relu @ W2 + b2
    ##your code ends here
    
    #return_intermediary_results should only be True if you are going to use this forward pass
    # to calculate gradients for the network parameters
    if return_intermediary_results:
        #if you are doing the forward pass to calculate backward pass afterwards, you are going to need all intermediary results
        to_return = {'out_1': out_1, 'out_1_relu': out_1_relu, 'out_2': out_2}
    else:
        #if you are doing the forward pass just to get the output of the network, you only need the final result
        to_return = out_2
    return to_return

x_ex1_train , y_ex1_train = return_vars_for_later()
#test your function to make sure it is doing what is expected
validate_ex22(two_layer_network_forward, initialize_parameters_ex2, x_ex1_train)


# ** END exercise 2.2 **

# ---------------- exercise 2.4 ----------------
def mse_loss_backward(predicted, gt):
    ##your code starts here
    derivative_of_mse_loss_with_respect_to_predicted= 2 * (predicted - gt) / predicted.shape[0]  # Derivative of MSE loss
    ##your code ends here
    return derivative_of_mse_loss_with_respect_to_predicted

def two_layer_network_backward(inputs, parameters, gt, loss_backward):
    
    intermediary_results_in_forward = two_layer_network_forward(inputs, parameters, return_intermediary_results = True)

    out_1 = intermediary_results_in_forward['out_1'] 
    out_1_relu = intermediary_results_in_forward['out_1_relu'] 
    out_2 = intermediary_results_in_forward['out_2'] 
    
    derivative_of_loss_with_respect_to_out_2 = loss_backward(out_2, gt) 
    
    ##your code starts here

    derivative_of_loss_with_respect_to_weights_2 = out_1_relu.T @ derivative_of_loss_with_respect_to_out_2
    derivative_of_loss_with_respect_to_bias_2 = np.sum(derivative_of_loss_with_respect_to_out_2, axis=0)

    dL_dout1_relu = derivative_of_loss_with_respect_to_out_2 @ parameters['weights_2'].T
    dL_dout1 = dL_dout1_relu * (out_1 > 0)  # ReLU derivative

    derivative_of_loss_with_respect_to_weights_1 = inputs.T @ dL_dout1
    derivative_of_loss_with_respect_to_bias_1 = np.sum(dL_dout1, axis=0)


    ##your code ends here
    
    return {
            'weights_1': derivative_of_loss_with_respect_to_weights_1,
            'bias_1': derivative_of_loss_with_respect_to_bias_1, 
            'weights_2':derivative_of_loss_with_respect_to_weights_2 , 
            'bias_2':derivative_of_loss_with_respect_to_bias_2
            }

def two_layer_network_mse_backward(inputs, parameters, gt):
    return two_layer_network_backward(inputs, parameters, gt, mse_loss_backward)



#test your function to make sure it is doing what is expected
test_gradient(two_layer_network_forward, two_layer_network_mse_backward, mse, x_ex1_train[0:20,:], y_ex1_train[0:20,:], initialize_parameters_ex2(1, 10, 1) )


# ** END exercise 2.4 **

# ---------------- exercise 2.5 ----------------
def run_batch_sgd(backward_function, parameters, learning_rate, inputs, targets):
    #calculate gradients and update parameters using sgd update rule
    ##your code starts here
    gradients = backward_function(inputs, parameters, targets)

    for param in parameters:
        parameters[param] -= learning_rate * gradients[param]  # Update rule

    return parameters
    ##your code ends here

n_hidden_nodes = 50
parameters_two_layer_regression = initialize_parameters_ex2(1, n_hidden_nodes, 1)

learning_rate = 0.001
batch_size = 1
n_epochs = 1000

for epoch in range(n_epochs):
    shuffled_indexes = (np.arange(x_ex1_train.shape[0]))
    np.random.shuffle(shuffled_indexes)
    shuffled_indexes = np.array_split(shuffled_indexes,x_ex1_train.shape[0]//batch_size )
    for batch_i in range(len(shuffled_indexes)):
        batch = shuffled_indexes[batch_i]
        input_this_batch = x_ex1_train[batch,:]
        gt_this_batch =  y_ex1_train[batch,:]
        #use you function run_batch_sgd to update the parameters
        ##your code starts here
        parameters_two_layer_regression = run_batch_sgd(
            two_layer_network_mse_backward,
            parameters_two_layer_regression,
            learning_rate,
            input_this_batch,
            gt_this_batch
        )
        ##your code ends here

#plot the results of training
##your code starts here
y_pred_train = two_layer_network_forward(x_ex1_train, parameters_two_layer_regression)

plt.figure(figsize=(8, 5))
plt.scatter(x_ex1_train, y_ex1_train, label='Training Data', color='blue')
plt.plot(x_ex1_train, y_pred_train, label='Neural Network Output', color='red')
plt.legend()
plt.savefig("exercise_2_5_plot.png")
plt.show()

##your code ends here


# ** END exercise 2.5 **