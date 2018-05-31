#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:42:12 2018

@author: alok
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('titanic_train.csv')
del data['PassengerId']
del data['Name']
del data['Ticket']
del data['Cabin']

le =    LabelEncoder()
data.Sex = le.fit_transform(data.Sex)
data.Age = data.Age.fillna(data.Age.mean())
data.Embarked = data.Embarked.fillna(data.Embarked.mode())
data.Embarked = le.fit_transform(data.Embarked)
#data.Embarked = data.Embarked.mode()

x= data.iloc[:,1:].values
y = data.iloc[:,0:1].values
#scaler = StandardScaler()
#x = scaler.fit_transform(x)
w = np.zeros((7,1))
 
b = 1
 
log = np.log10
 
iter = 1000
l = [0.0001,0.001]
cost = np.zeros(iter) 

#l = 0.0001
for item in l:
    l = item
    for i in range(iter):
        
     
        z = x.dot(w) + b 
        yh  = 1/(1+math.e**(-z))
         
        dz = yh -y 
        dw = (dz.T).dot(x) / len(x) 
           
        db = np.sum(dz) / len(x)   
        b= b -  db* l
        
        w = w - dw.T * l
         
        cost[i] = - sum(y * log(yh) + (1-y)  * log(1-yh))/len(x)
     
    #print(cost)
     
    plt.plot(np.arange(iter),cost)
#sum((yh-y.mean())**2)/sum((y-y.mean())**2)




  
