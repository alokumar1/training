#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 18:19:15 2018

@author: alok
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
d = pd.read_csv('data.txt',names = ['x','y'])
x = (d['x'].values).reshape(97,1)
y = (d['y'].values).reshape(97,1)

one = np.ones((97,1))
X = np.hstack((one,x))
a0,a1 = np.dot(np.dot(np.asmatrix(np.dot(X.transpose(), X)).I, X.transpose()),y)


Y = a0[0,0] + a1[0,0] *x
plt.plot(x,y,'ro')
plt.plot(x,Y)