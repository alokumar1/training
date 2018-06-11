#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:35:19 2018

@author: alok
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 21:37:02 2018

@author: alok
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 18:42:01 2018

@author: alok
"""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")
train_x = mnist.train.images
train_y = mnist.train.labels
train_x = train_x.reshape(55000,28,28)

train_x = train_x.reshape((55000,28,28,1))

def zero_pad(X, pad):   
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=(1,1))
    return X_pad

def conv_single_step(a_slice_prev, W, b):   
    s = a_slice_prev*W
    Z = np.sum(s)   
    Z = Z+b
    return Z

def conv_forward(A_prev, W, b, hparameters):
   
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
   
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    n_H =int((n_H_prev+2*pad -f)/stride + 1)
    n_W = int((n_W_prev+2*pad-f)/stride + 1)
    
    Z = np.zeros((m,n_H,n_W,n_C))
    
    A_prev_pad = zero_pad(A_prev,pad)
    
    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]
        
       
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                     
                    # Find the corners of the current "slice" 
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev,W[:,:,:,c],b[:,:,:,c])
                                        
    
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache

def pool_forward(A_prev, hparameters, mode = "max"):
   
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):                        
        for h in range(n_H):                    
            for w in range(n_W):                
                for c in range (n_C):            
                    
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    
    cache = (A_prev, hparameters)
    
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache
def batchnorm_forward(x, gamma, beta, eps):

  N, D = x.shape

  #step1: calculate mean
  mu = 1./N * np.sum(x, axis = 0)

  #step2: subtract mean vector of every trainings example
  xmu = x - mu

  #step3: following the lower branch - calculation denominator
  sq = xmu ** 2

  #step4: calculate variance
  var = 1./N * np.sum(sq, axis = 0)

  #step5: add eps for numerical stability, then sqrt
  sqrtvar = np.sqrt(var + eps)

  #step6: invert sqrtwar
  ivar = 1./sqrtvar

  #step7: execute normalization
  xhat = xmu * ivar

  #step8: Nor the two transformation steps
  gammax = gamma * xhat

  #step9
  out = gammax + beta

  #store intermediate
  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache


def batchnorm_backward(dout, cache):

  #unfold the variables stored in cache
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache

  #get the dimensions of the input/output
  N,D = dout.shape

  #step9
  dbeta = np.sum(dout, axis=0)
  dgammax = dout #not necessary, but more understandable

  #step8
  dgamma = np.sum(dgammax*xhat, axis=0)
  dxhat = dgammax * gamma

  #step7
  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar

  #step6
  dsqrtvar = -1. /(sqrtvar**2) * divar

  #step5
  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

  #step4
  dsq = 1. /N * np.ones((N,D)) * dvar

  #step3
  dxmu2 = 2 * xmu * dsq

  #step2
  dx1 = (dxmu1 + dxmu2)
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

  #step1
  dx2 = 1. /N * np.ones((N,D)) * dmu

  #step0
  dx = dx1 + dx2

  return dx, dgamma, dbeta

def conv_backward(dZ, cache):
    
    (A_prev, W, b, hparameters) = cache
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    (m, n_H, n_W, n_C) = dZ.shape
    
    dA_prev = np.zeros_like(A_prev)                           
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    A_prev_pad = zero_pad(A_prev,pad)
    dA_prev_pad = zero_pad(dA_prev,pad)
    
    for i in range(m):                      
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):                   
            for w in range(n_W):               
                for c in range(n_C):           
                    
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i,h,w,c]
                    dW[:,:,:,c] += a_slice*dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]
                    
        #dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]    
    
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db

def create_mask_from_window(x):

    mask = (x == np.max(x))
    
    
    return mask

def distribute_value(dz, shape):
   
    (n_H, n_W) = shape
    
    average = dz
    
    a = average*np.ones((n_H,n_W))/(n_H*n_W)
    
    return a


def pool_backward(dA, cache, mode = "max"):
   
    (A_prev, hparameters_pool) = cache
    
    stride = hparameters_pool['stride']
    f = hparameters_pool['f']
    
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    dA_prev = np.zeros_like(A_prev)
    
    for i in range(m):                       # loop over the training examples
        
       
        a_prev = A_prev[i]
        
        for h in range(n_H):                  
            for w in range(n_W):               
                for c in range(n_C):          
                    
                    vert_start = h *stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f
                    
                    if mode == "max":
                        
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                        
                    elif mode == "average":
                        
                        da = dA[i,h,w,c]
                        shape = (f,f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da,shape)
                        
    
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev

A_prev = train_x[0:1000]

#batchnorm
#mean = np.sum(A_prev,axis = 0)/len(A_prev)
#deviation = np.std(A_prev,axis = 0)
#scaled = (A_prev-mean)/deviation
#for i in range(1000):
#    for j in range(28):
#        for k in range(28):
#            if(np.isnan(scaled[i,j,k])):                
#                scaled[i,j,k] = 0

#A_prev = scaled

y = train_y[0:1000]
W = np.random.randn(3,3,1,3)/10
b = np.zeros((1,1,1,3))
W1 = np.random.randn(15,507)/10
b1 = np.zeros((15,1))
W2 = np.random.randn(10,15)/10
b2 = np.zeros((10,1))
l1,l2,l3,l4 = 0.001,0.001,0.001,0.0001

loss= []


hparameters = {"pad" : 0,
               "stride": 1}
hparameters_pool = {"stride" : 2, "f": 2}
gamma = 1
beta = 0
eps = 0.0000001

iters = 50
for x in range(iters):
    #first convoluted layer
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    
    
    zero_matrix = np.zeros_like(Z)
    
    #apply relu activation
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            for k in range(len(Z[0][0])):
                for c in range(len(Z[0][0][0])):
                    Z[i,j,k,c] = max(zero_matrix[i,j,k,c],Z[i,j,k,c])
                    
    #Z is activated matrix after 1st convolution                
    A = Z                    
    #A_prev = A           
    A_pool, cache_pool = pool_forward(A, hparameters_pool,mode = 'max')    
    
    #fully connected layer
    FC = A_pool.reshape((1000,A_pool[0].size))
    
    #applying batch norm
    #    mean =  np.sum(FC,axis = 0)/len(FC)
    #    dev = np.std(FC,axis = 0)    
    #    FC = (FC-mean)/dev
    #    for i in range(1000):
    #        for j in range(507):
    #            if(np.isnan(FC[i,j])):
    #                FC[i,j] = 0
    
    batch_out, cache_batch = batchnorm_forward(FC,gamma,beta,eps)
    
    
    
    
    
    Z1 = np.dot(batch_out,W1.T) + b1.T
    A1 = 1/(1+np.exp(-Z1))  
    
    Z2 = np.dot(A1,W2.T) + b2.T
    #A2 = 1/(1+np.exp(-Z2))
    
    #softmax 
    Z2 = np.exp(Z2)
    sums =  np.sum(Z2,axis = 1)
    sums = sums.reshape((1000,1))
    A2 = Z2/sums
    
    #one hot encoder
    Y = np.eye(1000,10)[y.reshape(-1)]
    loss.append(-sum(np.sum(Y*np.log(A2),axis = 1))/len(Y))
    print(datetime.now().minute)
    print(loss[x])
#    print(loss)
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2.T , A1)
    db2 = np.sum(dZ2,axis= 0,keepdims = True)
    
    W2 = W2 - l3 * dW2
    b2 = b2 - l3 * db2.T
    
    dZ1 = np.dot(dZ2,W2) * ((1-A1)*A1)
    dW1 = np.dot(dZ1.T , FC)
    db1 = np.sum(dZ1,axis= 0,keepdims = True)
    
    W1 = W1 - l2 * dW1
    b1 = b1 - l2 * db1.T
    
    dA0 = np.dot(dZ1,W1)
    #dA0 = dA0.reshape((1000,13,13,3))
    
    dA_batch, dgamma, dbeta = batchnorm_backward(dA0,cache_batch)
    gamma = gamma - l4 * dgamma
    beta = beta - l4 * dbeta
    
    dA_batch = dA_batch.reshape((1000,13,13,3))
    dA_conv = pool_backward(dA_batch ,cache_pool,mode = 'max')
    
    dA_prev, dW, db = conv_backward(dA_conv,cache_conv)
    
    W = W - l1 * dW
    b = b - l1 * db



#calculation of accuracy
maxx = np.max(A2 ,axis=1).reshape(1000,1)
A2 = A2 - maxx
for i in range(1000):
    for j in range(10):
        if(A2[i][j] == 0):
            A2[i][j] = 1
        else:
            A2[i][j] = 0
            
            
mat = np.arange(10).reshape(10,1)
p = np.dot(A2,mat)

cnt =0
for i in range(1000):
    if((y[i]-p[i]) == 0):
        cnt = cnt +1
acc = cnt/1000.0
print(acc)
plt.plot(np.arange(iters),loss)

