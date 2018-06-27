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
import time
from datetime import timedelta
#height x width = 299 x 299 pixels

module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
height, width = hub.get_expected_image_size(module)

data = pd.read_csv('labels.csv')
names = data.id.values
start = time.time()
Image_mod = tf.placeholder(tf.float32,shape = [None,299,299,3])
logits = module(Image_mod)
X = np.zeros((10222,2048))
init = tf.global_variables_initializer()
for i in range(269):
    start_loop = time.time()
    image = []
    for j in range(38):
#        x = x_raw.namelist()[i*38 + 1 + j]
#        img = cv2.imread(join(direc,x))
        img = cv2.imread(names[i*38 + 1 + j]+'.jpg')
        img = cv2.resize(img,(299,299))
        img = img/255.0
        image.append(img)
    array_image = np.array(image)
    x_t = 0
    with tf.Session() as sess:
        sess.run(init)
        x_t = sess.run(logits,feed_dict = {Image_mod : image})
    X[i*38:i*38+38,:] = x_t
    print(i)
    stop_loop = time.time()
    print(stop_loop - start_loop)   

stop = time.time()
print(stop - start)

#import pickle
#pickle.dump(X, open('Inception_v3.p', 'w'))

import pickle              # import module first
trained = pickle.load(open('Inception_logits.p', 'rb'))    
        
data = pd.read_csv('labels.csv')
y = data.breed.values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, criterion='gini', 
                             max_depth=100, min_samples_split=10, 
                             min_samples_leaf=5, min_weight_fraction_leaf=0.0, 
                             max_features='auto', max_leaf_nodes=None, 
                             min_impurity_decrease=0.0, min_impurity_split=None, 
                             bootstrap=True, oob_score=False, n_jobs=1,
                             random_state=None, verbose=0, warm_start=False,
                             class_weight=None)



start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()
time_dif = end_time - start_time
print(" Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
clf.score(X_train,y_train)
clf.score(X_test,y_test )
#99,77.1 with tuning




from sklearn import svm
#clf = svm.SVC()
clf = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=5, gamma=0.0005, kernel='rbf',
    max_iter=-1, probability=False, random_state=10, shrinking=True,
    tol=0.001, verbose=False)

start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()
time_dif = end_time - start_time
print(" Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
clf.score(X_train,y_train)
clf.score(X_test,y_test )
#82,77.9 without tuning
#76.6,74.7   with tuning and sigmoid kernel
#82.6,77.9  tuning 'ovo','rbf'
#98.14, 80.34   tuning c=10,gamma = 0.001,rbf
# 95.5,80.6    gamma = 0.0005



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model = XGBClassifier()
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
time_dif = end_time - start_time
print(" Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
model.score(X_train,y_train)
model.score(X_test,y_test )
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
#99.9,77.0



                                    






