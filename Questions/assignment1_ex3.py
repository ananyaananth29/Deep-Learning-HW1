#!/usr/bin/env python
# EXERCISE 3

import os

os.system('pip3 install matplotlib')
os.system('pip3 install numpy')
os.system('pip3 install pandas')
os.system('pip3 install python-mnist')
os.chdir("/uufs/chpc.utah.edu/common/home/u1520797/assignment1package")

import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from utils import *
from assignment1_ex2 import initialize_parameters_ex2, run_batch_sgd, two_layer_network_forward, two_layer_network_backward
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


x_ex3_train, x_ex3_val, x_ex3_test, y_ex3_train, y_ex3_val, y_ex3_test = load_and_preprocess_mnist()

#sanity check to see that data is as it is supposed to be
#plt.imshow(x_ex3_train[1000,:].reshape(28,28), cmap = 'Greys')
plt.imsave('sample_image.png', x_ex3_train[1000, :].reshape(28, 28), cmap='Greys')

print(y_ex3_train[1000])

# ---------------- exercise 3.1 ----------------
def l2_regularization_backward(inputs, parameters, gt):
    gradients = {}
    for parameter_name in parameters.keys():
        if 'weights' in parameter_name:
            # complete the equation to calculate the l2 regularization loss gradient for weights
            ##your code starts here
            gradients[parameter_name] = 2 * 0.01 * parameters[parameter_name]
            ##your code ends here
        elif 'bias' in parameter_name:
            # complete the equation to calculate the l2 regularization loss gradient for bias.
            # Remember, the L2 regularization loss for bias is 0.
            ##your code starts here
            gradients[parameter_name] = np.zeros_like(parameters[parameter_name])
            ##your code ends here
    return gradients

#a softmax calculation with numerical stability tricks
def softmax(logits, axis):
    # subtracting the maximum logit from all logits for each example and prevents overflow 
    # of the exponential function of the logits and does not change results of the softmax
    # because of properties of division of exponentials
    stabilizing_logits = logits - np.expand_dims(np.max(logits, axis = axis), axis = axis)
    
    # clipping all logits to a minimum of -10 prevents underflow of the exponentials and 
    # only changes the result of the softmax minimally, since we know that one logit has value 0
    # and exp^0>>exp(-10)
    stabilizing_logits = np.clip(stabilizing_logits, -10, None)
    
    #using the softmax classic equation, but with the modified logits to prevent numerical errors
    return np.exp(stabilizing_logits)/np.expand_dims(np.sum(np.exp(stabilizing_logits), axis = axis), axis = axis)

# a forward function combined the two-layer network and the softmax
def two_layer_network_softmax_forward(inputs, parameters):
    logits = two_layer_network_forward(inputs, parameters)
    return softmax(logits, axis = 1)

# a forward function combined the two-layer network and the softmax
def softmax_plus_ce_loss_backward(predicted, gt):
    #the derivative of the output of softmax function followed by a cross-entropy loss
    # with respect to the input is a beautifully simple equation equals to the softmax
    # of the inputs minus the one-hot encoded groundtruth
    return (softmax(predicted, axis = 1) - gt)/predicted.shape[0]

#the calculation of the gradient for the classification network
def two_layer_network_softmax_ce_backward(inputs, parameters, gt):
    return two_layer_network_backward(inputs, parameters, gt, softmax_plus_ce_loss_backward)

# a function to get how many logits predicted the right class when compared to gt
def count_correct_predictions(logits, gt):
    predicted_labels = one_hot(np.argmax(logits, axis = 1), logits.shape[1])
    return np.sum(np.logical_and(predicted_labels,gt))

def two_layer_network_ce_and_l2_regularization_backward(inputs, parameters, gt, regularization_multiplier):
    gradients = {}
    gradients1 = two_layer_network_softmax_ce_backward(inputs, parameters, gt)
    gradients2 = l2_regularization_backward(inputs, parameters, gt)
    for parameter_name in parameters:
        gradients[parameter_name] = gradients1[parameter_name] + regularization_multiplier * gradients2[parameter_name]
    return gradients

# ** END exercise 3.1 **

# ---------------- exercise 3.2 ----------------
n_hidden_nodes = 200  # Keeping this unchanged

##your code starts here
lr = 0.09
batch_sz = 64
num_epochs = 600
reg_strength = 0.05

model_params = initialize_parameters_ex2(x_ex3_train.shape[1], n_hidden_nodes, 10)

for epoch in range(num_epochs):
    shuffled_indices = np.arange(x_ex3_train.shape[0])
    np.random.shuffle(shuffled_indices)
    mini_batches = np.array_split(shuffled_indices, x_ex3_train.shape[0] // batch_sz)
    
    for mini_batch in mini_batches:
        batch_inputs = x_ex3_train[mini_batch, :]
        batch_targets = y_ex3_train[mini_batch, :]
        
        model_params = run_batch_sgd(
            lambda inputs, params, targets: two_layer_network_ce_and_l2_regularization_backward(
                inputs, params, targets, reg_strength
            ),
            model_params,
            lr,
            batch_inputs,
            batch_targets
        )
    
    if epoch % 5 == 0 or epoch == num_epochs - 1:
        val_logits = two_layer_network_softmax_forward(x_ex3_val, model_params)
        val_acc = count_correct_predictions(val_logits, y_ex3_val) / x_ex3_val.shape[0] * 100
        print(f"Epoch {epoch + 1}/{num_epochs} - Validation Accuracy: {val_acc:.2f}%")

final_val_logits = two_layer_network_softmax_forward(x_ex3_val, model_params)
final_accuracy = count_correct_predictions(final_val_logits, y_ex3_val) / x_ex3_val.shape[0] * 100
print(f"\nFinal Validation Accuracy: {final_accuracy:.3f}%")

test_logits = two_layer_network_softmax_forward(x_ex3_test, model_params)
test_accuracy = count_correct_predictions(test_logits, y_ex3_test) / x_ex3_test.shape[0] * 100
print(f"Test Accuracy: {test_accuracy:.3f}%")

saved_model_params = {0.001: model_params}
##your code ends here

# ** END exercise 3.2 **

# ---------------- exercise 3.3 ----------------
shuffled_indexes = (np.arange(x_ex3_test.shape[0]))
shuffled_indexes = np.array_split(shuffled_indexes,x_ex3_test.shape[0]//batch_size )
corrects = 0
total = 0
for batch_i in range(len(shuffled_indexes)):
    batch = shuffled_indexes[batch_i]
    corrects += count_correct_predictions(two_layer_network_forward(x_ex3_test[batch], parameters_two_layer_classification_ex3_dic[0.001]), y_ex3_test[batch])
    total += len(batch)
print('Test accuracy = ' + str(corrects/float(total)*100) + '%')

# ** END exercise 3.3 **