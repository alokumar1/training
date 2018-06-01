#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:44:36 2018

@author: alok
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('winequality-red.csv',delimiter=';' )
x = data.iloc[:,:11].values
y = data.iloc[:,11:12].values
scaler = StandardScaler()
x = scaler.fit_transform(x)
log = np.log10 

def calc_cost(x,y,b,w,iter,l):
    cost = np.zeros(iter) 
    for i in range(iter):
        yh = x.dot(w) + b          
        dz = yh -y         
        dw = (dz.T).dot(x) / len(x)            
        db = np.sum(dz) / len(x)   
        b= b -  db* l        
        w = w - dw.T * l        
        cost[i] = sum(dz**2)/len(x)
    return cost,b,w

def predict(b,w,x):
    return x.dot(w) + b

def calc_r_square(b,w,x,y):
    yh = predict(b,w,x)
    return  (sum((yh-y.mean())**2)/sum((y-y.mean())**2))
    
    
iter = 10000
l = [0.0001,0.001,0.01,0.1]
for item in l:
    l = item
    w = np.zeros((11,1))
    b = 0 
    cost,b,w = calc_cost(x,y,b,w,iter,l)        
    r_square = calc_r_square(b,w,x,y)
    print('r_square = '+ str(r_square)+ ' in case of learning rate = '+ str(l))
    plt.plot(np.arange(iter),cost,label = str(l))
    
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.legend()
plt.show() 



l = 0.0001
iterations = [10,100,1000,10000,100000]
for item in iterations:
    iter = item
    w = np.zeros((11,1))
    b = 0 
    cost,b,w = calc_cost(x,y,b,w,iter,l)        
    r_square = calc_r_square(b,w,x,y)
    print('r_square = '+ str(r_square)+ ' in case of iterations = '+ str(iter))
    plt.plot(np.arange(iter),cost,label = str(iter))
    
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.legend()
plt.show() 





    
