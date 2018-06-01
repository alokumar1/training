#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:42:12 2018

@author: alok
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def preprocess(data):    
    del data['PassengerId']
    del data['Name']
    del data['Ticket']
    del data['Cabin']
    le = LabelEncoder()
    data.Sex = le.fit_transform(data.Sex)
    data.Age = data.Age.fillna(data.Age.mean())
    data.Embarked = data.Embarked.fillna(data.Embarked.mode())
    data.Embarked = le.fit_transform(data.Embarked)
    return data

def split_data(train):    
    train = preprocess(train)
    x= train.iloc[:,1:].values
    y = train.iloc[:,0:1].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x, x_test, y, y_test = train_test_split(x,y,train_size = 0.80)
    return x,x_test,y,y_test 

def calc_cost(x,y,w,b,iter,l):
    cost = np.zeros(iter)
    for i in range(iter): 
        z = x.dot(w) + b    
        yh  = calc_sigmoid(z)   
        db,dw = calc_grad(y,yh,x,b)
        b= b -  db* l    
        w = w - dw.T * l   
        cost[i] = - sum(y * log(yh) + (1-y)  * log(1-yh))/len(x)
    return cost,w,b

def calc_sigmoid(z):
     return 1/(1+np.exp(-z)) 
 
def calc_grad(y,yh,x,b):
    dz = yh -y 
    dw = (dz.T).dot(x) / len(x)        
    db = np.sum(dz) / len(x)   
    return db,dw
    
def predict(w,b,x_test):
    z_pred = x_test.dot(w)+b
    y_pred = calc_sigmoid(z_pred) 
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    return y_pred    

def confusion_mat(y_test,y_pred):
    [[TN,FP],[FN,TP]] = confusion_matrix(y_test,y_pred)
    #accuracy = (y_test==y_pred).sum()/len(y_test)
    accuracy = float((TP+TN))/(TN+FP+FN+TP) 
    precision = float(TP)/(FP+TP)
    return accuracy,precision




train = pd.read_csv('titanic_train.csv')
x,x_test,y,y_test = split_data(train)    
log = np.log10


iter = 1000
alpha = [0.001,0.01,0.1,1,2,5,10]
for item in alpha:
    l = item
    w = np.zeros((7,1))
    b = 0
    cost,w,b = calc_cost(x,y,w,b,iter,l) 
    plt.plot(np.arange(iter),cost,label= 'rate '+str(l))
    y_pred = predict(w,b,x_test)
    accuracy ,precision = confusion_mat(y_test,y_pred)
    print('accuracy = ' + str(accuracy)+' in case of learning rate = ' + str(item))   
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.legend()
plt.show()  


iterations= [10,100,1000,10000]
l= 0.1
for item in iterations:
    iter = item
    w = np.zeros((7,1))
    b = 0
    cost,w,b = calc_cost(x,y,w,b,iter,l) 
    plt.plot(np.arange(iter),cost,label= 'iter ' + str(item))
    y_pred = predict(w,b,x_test)
    accuracy ,precision = confusion_mat(y_test,y_pred)
    print('accuracy = ' + str(accuracy)+' in case of iterations = ' + str(item) )   
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.legend()
plt.show()  


#test = pd.read_csv('titanic_test.csv')
#test.isnull().sum()
#test = preprocess(test)
#test = test.fillna(test.mean())
#test = test.values
#scaler = StandardScaler()
#test = scaler.fit_transform(test)
#z_pred = test.dot(w)+b
#y_pred  = 1/(1+np.exp(-z_pred))
#y_pred[y_pred>0.5] = 1
#y_pred[y_pred<0.5] = 0












  
