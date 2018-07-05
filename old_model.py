#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 12:39:36 2018

@author: alok
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import math
import pickle

data = pickle.load(open('/home/alok/spyder/plant seedlings/4750_images.p','rb'))
classes = pickle.load(open('/home/alok/spyder/plant seedlings/4750_classes.p','rb'))
train_x,test_x,train_y,test_y=train_test_split(data,classes,test_size = 0.20)  
del data,classes

#enc = LabelEncoder()
#classes = enc.fit_transform(classes)
#classes = classes.reshape(4750,1)
#henc = OneHotEncoder()
#classes = henc.fit_transform(classes).toarray()
#classes = classes.astype(np.float32)
#classes = np.array(classes)
#train_x,test_x,train_y,test_y=train_test_split(data,classes,test_size = 0.20)  
#del data,classes



x = tf.placeholder(tf.float32, shape=[None, 64,64,3], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, 12], name='y_true_placeholder')
y_true_cls = tf.argmax(y_true, dimension=1,name='argmax')

def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
    
    with tf.variable_scope(name) as scope:
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        layer += biases
        
        return layer, weights
    
def new_pool_layer(input, name):
    
    with tf.variable_scope(name) as scope:
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        return layer
    
    
def new_relu_layer(input, name):
    
    with tf.variable_scope(name) as scope:
        layer = tf.nn.relu(input)
        
        return layer
def new_fc_layer(input, num_inputs, num_outputs, name):
    
    with tf.variable_scope(name) as scope:

        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        
        layer = tf.matmul(input, weights) + biases
        
        return layer
    
def random_mini_batches(X, Y, mini_batch_size ):
   m = X.shape[0]                 
   mini_batches = []   
   # Shuffle (X, Y)
   permutation = list(np.random.permutation(m))
   shuffled_X = X[permutation,:]
   shuffled_Y = Y[permutation,:]

   # Partition
   num_complete_minibatches = int(math.floor(m/mini_batch_size) )
   for k in range(0, num_complete_minibatches):
       mini_batch_X = shuffled_X[ k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
       mini_batch_Y = shuffled_Y[ k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
       mini_batch = (mini_batch_X, mini_batch_Y)
       mini_batches.append(mini_batch)
   
   # Handling the end case (last mini-batch < mini_batch_size)
   if m % mini_batch_size != 0:
       mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
       mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
       mini_batch = (mini_batch_X, mini_batch_Y)
       mini_batches.append(mini_batch)
   
   return mini_batches

    
# Convolutional Layer 1
layer_conv1, weights_conv1 = new_conv_layer(input=  x, num_input_channels=3, filter_size=5, num_filters=8, name ="conv1")

# Pooling Layer 1
layer_pool1 = new_pool_layer(layer_conv1, name="pool1")

# RelU layer 1
layer_relu1 = new_relu_layer(layer_pool1, name="relu1")

drop_layer1 = tf.nn.dropout(layer_relu1,keep_prob=0.8,name='drop1')

norm_layer1 = tf.contrib.layers.batch_norm(drop_layer1)



# Convolutional Layer 2
layer_conv2, weights_conv2 = new_conv_layer(input=norm_layer1, num_input_channels=8, filter_size=5, num_filters=16, name= "conv2")

# Pooling Layer 2
layer_pool2 = new_pool_layer(layer_conv2, name="pool2")

# RelU layer 2
layer_relu2 = new_relu_layer(layer_pool2, name="relu2")

drop_layer2 = tf.nn.dropout(layer_relu2,keep_prob=0.8,name='drop2')


norm_layer2 = tf.contrib.layers.batch_norm(drop_layer2)



layer_conv3, weights_conv3 = new_conv_layer(input=norm_layer2, num_input_channels=16, filter_size=5, num_filters=32, name= "conv3")

# Pooling Layer 2
layer_pool3 = new_pool_layer(layer_conv3, name="pool3")

# RelU layer 2
layer_relu3 = new_relu_layer(layer_pool3, name="relu3")

drop_layer3 = tf.nn.dropout(layer_relu3,keep_prob=0.8,name='drop3')


norm_layer3 = tf.contrib.layers.batch_norm(drop_layer3)


layer_conv4, weights_conv4 = new_conv_layer(input=norm_layer3, num_input_channels=32, filter_size=5, num_filters=64, name= "conv4")

# Pooling Layer 2
layer_pool4 = new_pool_layer(layer_conv4, name="pool4")

# RelU layer 2
layer_relu4 = new_relu_layer(layer_pool4, name="relu4")

drop_layer4 = tf.nn.dropout(layer_relu4,keep_prob=0.8,name='drop4')


norm_layer4 = tf.contrib.layers.batch_norm(drop_layer4)


layer_conv5, weights_conv5 = new_conv_layer(input=norm_layer4, num_input_channels=64, filter_size=5, num_filters=128, name= "conv5")

# Pooling Layer 2
layer_pool5 = new_pool_layer(layer_conv5, name="pool5")

# RelU layer 2
layer_relu5 = new_relu_layer(layer_pool5, name="relu5")

drop_layer5 = tf.nn.dropout(layer_relu5,keep_prob=0.8,name='drop5')


norm_layer5 = tf.contrib.layers.batch_norm(drop_layer5)


layer_conv6, weights_conv6 = new_conv_layer(input=norm_layer5, num_input_channels=128, filter_size=5, num_filters=128, name= "conv6")

# Pooling Layer 2
layer_pool6 = new_pool_layer(layer_conv6, name="pool6")

# RelU layer 2
layer_relu6 = new_relu_layer(layer_pool6, name="relu6")

drop_layer6 = tf.nn.dropout(layer_relu6,keep_prob=0.8,name='drop6')


norm_layer6 = tf.contrib.layers.batch_norm(drop_layer6)



layer_conv7, weights_conv7 = new_conv_layer(input=norm_layer6, num_input_channels=128, filter_size=5, num_filters=128, name= "conv7")

# Pooling Layer 2
layer_pool7 = new_pool_layer(layer_conv7, name="pool7")

# RelU layer 2
layer_relu7 = new_relu_layer(layer_pool7, name="relu7")

drop_layer7 = tf.nn.dropout(layer_relu7, keep_prob=0.8,name='drop7')


norm_layer7 = tf.contrib.layers.batch_norm(drop_layer7)



layer_conv8, weights_conv8 = new_conv_layer(input=norm_layer7, num_input_channels=128, filter_size=5, num_filters=256, name= "conv8")

# Pooling Layer 2
layer_pool8 = new_pool_layer(layer_conv8, name="pool8")

# RelU layer 2
layer_relu8 = new_relu_layer(layer_pool8, name="relu8")

drop_layer8 = tf.nn.dropout(layer_relu8,keep_prob=0.8,name='drop8')


norm_layer8 = tf.contrib.layers.batch_norm(drop_layer8)


# Flatten Layer
num_features = norm_layer8.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(norm_layer8, [-1, num_features],name='flattened_layer')

# Fully-Connected Layer 1
layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")

drop_layer_fc1 = tf.nn.dropout(layer_fc1,keep_prob=0.8,name='dropfc1')

norm_layer_fc1 = tf.contrib.layers.batch_norm(drop_layer_fc1)
# RelU layer 3
layer_relu3 = new_relu_layer(norm_layer_fc1, name="relu3")

# Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=12, name="fc2")

with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(layer_fc2,name='pred_softmax')
    y_pred_cls = tf.argmax(y_pred, dimension=1,name='argmax_pred')
with tf.name_scope("cross_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#writer = tf.summary.FileWriter("/home/alok/graphs/1/")
#writer1 = tf.summary.FileWriter("/home/alok/graphs/2/")
#
#loss_summary = tf.summary.scalar('loss', cost)
#accuracy_summary = tf.summary.scalar('accuracy', accuracy)
#merged_summary = tf.summary.merge_all()
#merged_summary = tf.summary.merge([loss_summary,accuracy_summary])

#loss_summary_test = tf.summary.scalar('loss_test', cost)
#accuracy_summary_test = tf.summary.scalar('accuracy_test', accuracy)
#merged_summary2 = tf.summary.merge([loss_summary_test,accuracy_summary_test])
num_epochs = 2 
minibatch_size =32



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#    writer.add_graph(sess.graph)
#    writer1.add_graph(sess.graph)
    i = 0
    for epoch in range(num_epochs):
        
        start_time = time.time()
        train_accuracy = 0.
        vali_accuracy = 0.
        epoch_train_accuracy = 0.
        epoch_vali_accuracy = 0.
                
        num_minibatches = int(len(train_x) / minibatch_size)
        minibatches = random_mini_batches(train_x, train_y, minibatch_size)
        for minibatch in minibatches:
            (batch_x, batch_y) = minibatch
            feed_dict_train = {x: batch_x, y_true: batch_y}
    
            sess.run(optimizer, feed_dict=feed_dict_train)
#            summ,train_accuracy = sess.run([merged_summary,accuracy], feed_dict=feed_dict_train)           
#            writer.add_summary(summ,i) 
            
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict_train)
            epoch_train_accuracy += train_accuracy
            
#            summ, vali_accuracy = sess.run([merged_summary, accuracy], feed_dict={x:test_x, y_true:test_y})
#            writer.add_summary(summ, i)
            
            vali_accuracy = sess.run(accuracy, feed_dict={x:test_x, y_true:test_y})
            epoch_vali_accuracy += vali_accuracy
            
            print(str(epoch)+' epoch '+' iter '+str(i)+' train_accuracy: '+str(train_accuracy)+' test_acc: '+str(vali_accuracy))
            i+=1
        end_time = time.time()
        
        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
        print("\tAccuracy:")
        print ("\t- Training Accuracy:\t{}".format(epoch_train_accuracy/len(minibatches)))
        print ("\t- Validation Accuracy:\t{}".format(epoch_vali_accuracy/len(minibatches)))
        
        
#writer.close()
