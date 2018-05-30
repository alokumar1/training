#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:55:48 2018

@author: alok
"""

import pandas as pd
import numpy as np
data  = pd.read_csv('home.txt',names = np.arange(3))
data = (data - data.mean())/data.std()

x = data.iloc[:,:-1]
y = data.iloc[:,2:3]

ones = np.ones([x.shape[0],1])

x = np.concatenate((ones,x),axis = 1)
y = y.values

lmda = 0.1
alpha = 0.01


theta = np.zeros([1,3])



for i in range(1000):
    
    theta = theta*(1-(alpha*lmda)/len(x)) - (alpha/len(x)) * np.sum(x * (x* theta - y), axis=0)
print(theta)

