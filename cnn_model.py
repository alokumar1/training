#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:22:26 2018

@author: alok
"""
import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

def conv2d(x, W, b, strides=1,padding='VALID'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x,f = 3, k=2,padding = 'VALID'):
    return tf.nn.max_pool(x, ksize=[1, f, f, 1], strides=[1, k, k, 1],padding=padding)


def conv_net(x, weights, biases, dropout):

    conv1 = conv2d(x, weights['wc1'], biases['bc1'],strides=2)
    conv1 = tf.nn.relu(conv1)    

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'],strides=1) 
    conv2 = tf.nn.relu(conv2)    
    
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'],strides=1,padding='SAME')  
    conv3 = maxpool2d(conv3,f=3, k=2)
    conv3 = tf.nn.relu(conv3)  
    print(conv3.shape)
    
#    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'],strides=1)   
#    conv4 = tf.nn.relu(conv4)    
#    print(conv4.shape)
    
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)  
    print(fc1.shape)
    
#    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
#    fc2 = tf.nn.relu(fc2)
#    fc2 = tf.nn.dropout(fc2, dropout)    
    
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

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

#def main_func(x,y,learning_rate = 0.005, num_epochs = 5,minibatch_size = 1000):
num_classes = 12
height = 64
width = 64
learning_rate = 0.005
num_epochs = 100
minibatch_size = 100



weights = { 'wc1': tf.Variable(tf.random_normal([2, 2, 3, 32])),
            'wc2': tf.Variable(tf.random_normal([2, 2, 32, 32])),
            'wc3': tf.Variable(tf.random_normal([2,2,32, 64])),
#            'wc4': tf.Variable(tf.random_normal([14*14*64, 64])),
            'wd1': tf.Variable(tf.random_normal([15*15*64, 128])),
#            'wd2': tf.Variable(tf.random_normal([128, 256])),

            'out': tf.Variable(tf.random_normal([128, num_classes]))}

biases = {  'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([32])),
            'bc3': tf.Variable(tf.random_normal([64])),
#            'bc4': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([128])),
#            'bd2': tf.Variable(tf.random_normal([256])),
            'out': tf.Variable(tf.random_normal([num_classes]))}


acc = []
costs = []
test = []
train_x,test_x,train_y,test_y=train_test_split(data,classes,test_size = 0.20)  


X = tf.placeholder(tf.float32, [None, height,width,3])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) 
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)


correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()    
#        saver = tf.train.Saver(max_to_keep = 1)#    
#writer = tf.summary.FileWriter('/home/alok/spyder/plant seedlings/tensorboard/',tf.get_default_graph())
#tf.summary.scalar('accuracy',accuracy)   
#tf.summary.scalar('cost',loss_op)  
#merged = tf.summary.merge_all()
#           
with tf.Session() as sess:
#    print('started')
    sess.run(init)
#    print('initialized')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_cost = 0.
        epoch_accuracy = 0.
        epoch_test_accuracy = 0.
        num_minibatches = int(len(data) / minibatch_size)
        minibatches = random_mini_batches(train_x, train_y, minibatch_size)
        #running loop over different minibatches
        for minibatch in minibatches:
#            print('minibatches formed')
            (batch_x, batch_y) = minibatch
            
            #cost calculation
            _, minibatch_cost = sess.run([optimizer,loss_op],feed_dict={X:batch_x, Y:batch_y, keep_prob:0.7})
#            _, minibatch_cost,summ = sess.run([optimizer,loss_op,merged],feed_dict={X:batch_x, Y:batch_y, keep_prob:0.8})
#            print('calculated cost')
            epoch_cost += 0.001*minibatch_cost / num_minibatches
  
#            writer.add_summary(summ,epoch)                
            minibatch_acc = sess.run(accuracy, feed_dict={X: batch_x,Y:batch_y,keep_prob: 0.7})
            epoch_accuracy += minibatch_acc / num_minibatches
#            print('accuracy calculated')
            #test accuracy
            test_acc = sess.run(accuracy, feed_dict={X: test_x,Y:test_y,keep_prob: 0.7})
            epoch_test_accuracy += test_acc / num_minibatches
            
#            acc_cost,summm = sess.run([loss_op,merged],feed_dict={X:test_x, Y:test_y, keep_prob:1.0})
            
            
            print(str(epoch)+ ' cost: '+str(0.001*minibatch_cost)+' train_acc: '+str(minibatch_acc)+' test_acc: '+str(test_acc))
            
           
            
        costs.append(epoch_cost)
        acc.append(epoch_accuracy)
        test.append(epoch_test_accuracy)
        print(str(epoch)+' epoch,' +' epoch_cost: '+ str(epoch_cost)+', epoch_accuracy: '+str(epoch_accuracy)+'test_accuray: '+str(epoch_test_accuracy))         
       
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            
        
#            save after every 5th iteration
#            if(epoch %5 == 0):
#                saver.save(sess,save_path='/home/alok/spyder/checkpoints/breed/1/',global_step=epoch)
    
#    plt.plot(np.arange(len(costs)),costs)
#    plt.plot(np.arange(len(acc)),acc)
#    plt.plot(np.arange(len(test)),test)
      
    writer.close()
#        return costs,acc,test


#data = pd.read_csv('labels.csv')
#names = data.id
#dogs = []
#breed = []

##reading images
#for i in range(len(names)):
#    img = cv2.imread(names[i]+'.jpg')
#    img = cv2.resize(img,(81,81), interpolation = cv2.INTER_CUBIC)
#    dogs.append(img/255.0)
#    breed.append(data.breed[i]) 
#    
#dogs = np.array(dogs)
#breed = np.array(breed)

#encoding breed names
enc = LabelEncoder()
classes = enc.fit_transform(classes)
classes = classes.reshape(4750,1)
henc = OneHotEncoder()
classes = henc.fit_transform(classes).toarray()

#get cost , train and test accuracy
costs,acc,test= main_func(data,classes,learning_rate = 0.005, num_epochs = 50,minibatch_size = 1000)
     
