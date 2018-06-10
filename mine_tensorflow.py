#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:56:12 2018

@author: alok
"""

#import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")
import matplotlib.pyplot as plt
import numpy as np
#28*28
train_x = mnist.train.images
train_y = mnist.train.labels

#test_x = mnist.test.images
#test_y = mnist.test.labels

train_x = train_x.reshape(55000,28,28)
#test_x=  test_x.reshape(10000,28,28)

#for i in range(10):
#    plt.imshow(train_x[i], cmap='gray')
#    plt.show()
#    print(train_y[i])

x = train_x[2].T
y = train_y[2]
#pad in x gives padding in vertical direction
def zero_pad(mat,shape):
    x,y = shape
    padded = np.ones((mat.shape[0]+2*x, mat.shape[1]+2*y))
    padded[x:padded.shape[0]-x , y:padded.shape[1]-y] = mat
    
    plt.imshow(mat, cmap='gray')
    plt.imshow(padded, cmap='gray')
    #plt.show()    
    return padded


l = 0.01
f = 3
p = 0
s = 1
n_filt = 2
h = (x.shape[0] - f + 2*p)/s + (1)
w = (x.shape[1] - f + 2*p)/s + (1)
c = n_filt

empty_nxt_mat = np.zeros((h,w,c))

filtr = np.random.randn(f,f,c)

#reshaes 1,28,28 
x = np.dstack(x)
x = x.reshape(28,28,1)

#reshaping into 3d
#x = np.dstack((x,x))
bias = [0,0]
#convoluton 
for i in range(h):
    for j in range(w):
        z_sum = x[i:i+f,j:j+f,:]*filtr 
#        z_sum = x[0:3,0:3,:]*filtr
        empty_nxt_mat[i,j,:] =   np.sum(np.sum(z_sum,axis=0),axis = 0)             
#empty_nxt_mat[0,0,0] =   np.sum(x[0:3,0:3]*filtr[:,:,0])

for i in range(len(bias)):
    empty_nxt_mat[:,:,i] =  empty_nxt_mat[:,:,i] + bias[i]
    
    
empty_nxt_mat = empty_nxt_mat +bias
convoluted_mat = np.zeros_like(empty_nxt_mat)

#relu activation
for i in range(h):
    for j in range(w):
        for k in range(c):
            convoluted_mat[i,j,k] = max(convoluted_mat[i,j,k],empty_nxt_mat[i,j,k])

#sigmoid
#
#convoluted_mat = 1/(1+np.exp(-1* empty_nxt_mat))

f_pool = 2
s_pool = 2
h_pool = (convoluted_mat.shape[0] - f_pool )/s_pool + (1)
w_pool = (convoluted_mat.shape[1] - f_pool )/s_pool + (1)
c_pool = 2

pooled_mat = np.zeros((h_pool,w_pool,c_pool))

#####
mask_mat = convoluted_mat
#####

for i in range(h_pool):
    for j in range(w_pool):
        for k in range(c_pool):
            hs = i *s_pool 
            he = hs+f_pool
            vs = j*s_pool
            ve = vs+f_pool
            pooled_mat[i,j,k] = np.max(convoluted_mat[hs:he,vs:ve,k])
            
            #####
            mask_mat[hs:he,vs:ve,k] = (convoluted_mat[hs:he,vs:ve,k] == np.max(convoluted_mat[hs:he,vs:ve,k]))
            #####

pool_fc = pooled_mat.reshape((13*13*2 , 1))
A0 = pool_fc
#15 nodes
W1 = np.random.randn(15,pool_fc.size)
b1 = np.zeros((15,1))
Z1 = np.dot(W1,pool_fc)+b1
A1 = 1/(1+np.exp(-1* Z1))

W2 = np.random.randn(10,A1.size)
b2 = np.zeros((10,1))    
Z2 = np.dot(W2,A1)+b2
#A2 = 1/(1+np.exp(-1* Z2))

#softmax,yhat
A2 = np.exp(Z2)/np.sum(np.exp(Z2))

#y actual in vector
Y = np.zeros_like(A2)
Y[y] = 1

yh  = A2
loss = -sum(Y*np.log(yh))

#backprop in layer 2
dZ2= yh - y
dW2 = np.dot(dZ2, A1.T)
W2 = W2 - l * dW2 
db2 = dZ2
b2 = b2- l*db2

#backprop in layer 1
dZ1 = np.dot(W2.T , dZ2) * ((1-A1)*A1)
dW1 = np.dot(dZ1,A0.T)
db1 = dZ1
W1 = W1- l * dW1
b1 = b1 - l* db1

dA0 = np.dot(W1.T , dZ1)
back_pool = dA0.reshape(13,13,2)

def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    
    ### START CODE HERE ### (≈1 line)
    mask = (x == np.max(x))
    ### END CODE HERE ###
    
    return mask
def pool_backward(dA,A_prev, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    
    ### START CODE HERE ###
    
    # Retrieve information from cache (≈1 line)
    #(A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = 2
    f = 2
    
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
#    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    n_H_prev, n_W_prev, n_C_prev = A_prev.shape
#    m, n_H, n_W, n_C = dA.shape
    n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros_like(A_prev)
    
#    for i in range(m):                       # loop over the training examples
        
        # select training example from A_prev (≈1 line)
    a_prev = A_prev#[i]
        
    for h in range(n_H):                   # loop on the vertical axis
        for w in range(n_W):               # loop on the horizontal axis
            for c in range(n_C):           # loop over the channels (depth)
                
                # Find the corners of the current "slice" (≈4 lines)
                vert_start = h *stride
                vert_end = vert_start + f
                horiz_start = w*stride
                horiz_end = horiz_start + f
                
                # Compute the backward propagation in both modes.
                if mode == "max":
                    
                    # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                    a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                    # Create the mask from a_prev_slice (≈1 line)
                    mask = create_mask_from_window(a_prev_slice)
                    # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
#                    dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                    dA_prev[vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[h, w, c]

#                    elif mode == "average":
#                        
#                        # Get the value a from dA (≈1 line)
#                        da = dA[i,h,w,c]
#                        # Define the shape of the filter as fxf (≈1 line)
#                        shape = (f,f)
#                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
#                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da,shape)
                        
    ### END CODE ###
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev

dA_conv = pool_backward(back_pool,convoluted_mat,'max')

dZ_conv = dA_conv * ((1-convoluted_mat)*convoluted_mat)

dW = np.zeros_like(filtr)

for i in range(h):
    for j in range(w):
        
        vs = i *1
        ve = vs + 3
        hs = j*1
        he = hs + 3
        a_slice = x[vs:ve,hs:he,0]
        dW[:,:,0] += a_slice*dZ_conv[i,j,0]
        
for i in range(h):
    for j in range(w):
        
        vs = i *1
        ve = vs + 3
        hs = j*1
        he = hs + 3
        a_slice = x[vs:ve,hs:he,0]
        dW[:,:,1] += a_slice*dZ_conv[i,j,1]
        
filtr = filtr - l * dW


