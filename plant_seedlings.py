#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:48:54 2018

@author: alok
"""
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import pickle        
from sklearn.model_selection import train_test_split    
from sklearn import svm
from sklearn.metrics import confusion_matrix

labels = {0:'Black-grass',
          1:'Charlock',
          2:'Cleavers',
          3:'Common Chickweed',
          4:'Common wheat',
          5:'Fat Hen',
          6:'Loose Silky-bent',
          7:'Maize',
          8:'Scentless Mayweed',
          9:'Shepherds Purse',
          10:'Small-flowered Cranesbill',
          11:'Sugar beet'}

#dirctory = '/home/alok/spyder/plant seedlings/train/'+labels[0]


import os
import glob
classes = []
data = []
for val in labels:    
    img_dir = '/home/alok/spyder/plant seedlings/train/'+labels[val] # Enter Directory of all images 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)    
    for f1 in files:
        img = cv2.imread(f1)
        img = cv2.resize(img,(64,64))
        img =img/255.0
        data.append(img)
        classes.append(labels[val])
    print(labels[val] + ' done' )


data = np.array(data)
data = data.astype(np.float32)
data = np.array(data)
classes = np.array(classes)



def get_pretrained_features(data):    
    module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
    height, width = hub.get_expected_image_size(module)
    image_module = tf.placeholder(tf.float32,shape = [None,299,299,3])
    logits = module(image_module)
    final_features = np.zeros((4750,2048))
    init = tf.global_variables_initializer()
    for i in range(95):
        image = []
        for j in range(50):
            img = data[i*50+ j]
            img = img/255.0
            image.append(img)
        array_image = np.array(image)
        features = 0
        with tf.Session() as sess:
            sess.run(init)
            features = sess.run(logits,feed_dict = {image_module : array_image})
        final_features[i*50:i*50+50,:] = features
        print(i)
        pickle.dump(final_features, open('Inception3.p', 'w'))
    return final_features


def using_svm(X_train, X_test, y_train, y_test):
    clf = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=5, gamma=0.0005, kernel='rbf',
        max_iter=-1, probability=False, random_state=10, shrinking=True,
        tol=0.001, verbose=False)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train,y_train)
    test_acc = clf.score(X_test,y_test )
    return train_acc,test_acc
#95.4,87.4

final_features = get_pretrained_features(data)
X_train, X_test, y_train, y_test = train_test_split(final_features, classes, test_size=0.20, random_state=42)
train_acc,test_acc = using_svm(X_train, X_test, y_train, y_test)

