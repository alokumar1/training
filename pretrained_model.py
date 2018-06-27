#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 18:23:55 2018

@author: alok
"""

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import pandas as pd
import pickle        
from sklearn.cross_validation import train_test_split    
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def get_pretrained_features(data):    
    module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
    height, width = hub.get_expected_image_size(module)
    image_module = tf.placeholder(tf.float32,shape = [None,299,299,3])
    logits = module(image_module)
    final_features = np.zeros((10222,2048))
    names = data.id.values
    init = tf.global_variables_initializer()
    for i in range(269):
        image = []
        for j in range(38):
            img = cv2.imread(names[i*38 + 1 + j]+'.jpg')
            img = cv2.resize(img,(299,299))
            img = img/255.0
            image.append(img)
        array_image = np.array(image)
        features = 0
        with tf.Session() as sess:
            sess.run(init)
            features = sess.run(logits,feed_dict = {image_module : array_image})
        final_features[i*38:i*38+38,:] = features
#    pickle.dump(final_features, open('Inception3.p', 'w'))
    return final_features

def using_random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', 
                             max_depth=100, min_samples_split=10, 
                             min_samples_leaf=5, min_weight_fraction_leaf=0.0, 
                             max_features='auto', max_leaf_nodes=None, 
                             min_impurity_decrease=0.0, min_impurity_split=None, 
                             bootstrap=True, oob_score=False, n_jobs=1,
                             random_state=None, verbose=0, warm_start=False,
                             class_weight=None)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train,y_train)
    test_acc = clf.score(X_test,y_test )   
    return train_acc,test_acc
#99,77.1 with tuning

def using_XGBoost(X_train, X_test, y_train, y_test):    
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc=accuracy_score(y_test, y_pred)
    return test_acc
#99.9,77.0

def using_svm(X_train, X_test, y_train, y_test):
    clf = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=5, gamma=0.0005, kernel='rbf',
        max_iter=-1, probability=False, random_state=10, shrinking=True,
        tol=0.001, verbose=False)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train,y_train)
    test_acc = clf.score(X_test,y_test )
    return train_acc,test_acc
#82,77.9 without tuning
#76.6,74.7   with tuning and sigmoid kernel
#82.6,77.9  tuning 'ovo','rbf'
#98.14, 80.34   tuning c=10,gamma = 0.001,rbf
# 95.5,80.6    gamma = 0.0005


x = pickle.load(open('Inception_logits.p', 'rb'))       
data = pd.read_csv('labels.csv')
y = data.breed.values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
train_acc,test_acc = using_svm(X_train, X_test, y_train, y_test)


