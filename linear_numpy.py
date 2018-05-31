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

l = 0.01

w = np.zeros((11,1))
b = 1
 
log = np.log10
 
iter = 1000
l = [0.001,0.01,0.1,1]
cost = np.zeros(iter) 

for item in l:
    l = 0.01
    for i in range(iter):
        yh = x.dot(w) + b 
         
        dz = yh -y 
        
        dw = (dz.T).dot(x) / len(x) 
           
        db = np.sum(dz) / len(x)   
        b= b -  db* l
        
        w = w - dw.T * l
        
        cost[i] = sum(dz**2)/len(x)
        
    plt.plot(np.arange(iter),cost)
#sum((yh-y.mean())**2)/sum((y-y.mean())**2)