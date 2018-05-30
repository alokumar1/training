#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 14:12:56 2018

@author: alok
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
d = pd.read_csv('data.txt',names = ['x','y'])
x = d['x']
y = d['y']
a0 = 0
a1 = 0
l = 0.01

def cost_function(yh,y,d):
    return sum((yh-y)**2)/(2*d.shape[0])
c = []
cy =[]
prv_cost = 0
i = 0
while 1:
    yh = a0 + a1*x  
    cost = cost_function(yh,y,d)
    a0 = a0 -l* sum(yh-y)/d.shape[0]    
    a1 = a1 - l* sum((yh-y)*x)/d.shape[0]    
    c.append(cost)
    cy.append(i)
    i = i+1
    print('cost = '  +str(cost) + ' change in cost = ' +str( prv_cost  - cost), i)
    if(0< ( prv_cost  - cost)  < (10**-5)):
        break
    prv_cost = cost
    


    
print(a0, a1)
plt.plot(x,y,'o')
Y = a0+a1*x
plt.plot(x,Y)