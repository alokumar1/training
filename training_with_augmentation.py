#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:06:42 2018

@author: alok
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 15:37:58 2018

@author: alok
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:57:07 2018

@author: alok
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 22:26:13 2018

@author: alok
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 09:40:13 2018

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
from sklearn.model_selection import train_test_split

def conv2d(x, W, b, strides=1,padding='VALID'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x,f = 3, k=2,padding = 'VALID'):
    return tf.nn.max_pool(x, ksize=[1, f, f, 1], strides=[1, k, k, 1],padding='VALID')


def conv_net(x, weights, biases, dropout):

    conv1 = conv2d(x, weights['wc1'], biases['bc1'],strides=2)
    conv1 = tf.nn.relu(conv1)    

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'],strides=1) 
    conv2 = tf.nn.relu(conv2)    
    
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'],strides=1,padding='SAME')  
    conv3 = maxpool2d(conv3,f=3, k=2)
    conv3 = tf.nn.relu(conv3)    
    
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'],strides=1)   
    conv4 = tf.nn.relu(conv4)    
    
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)  
    
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)    
    
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
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

def main_func(x,y,learning_rate = 0.005,num_epochs = 30,minibatch_size = 300):
    num_classes = 120 
    height = 151
    width = 151
    
    weights = { 'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
                'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
                'wc3': tf.Variable(tf.random_normal([3, 3, 32, 64])),
                'wc4': tf.Variable(tf.random_normal([3, 3, 64, 128])),
                'wd1': tf.Variable(tf.random_normal([33*33*128, 256])),
                'wd2': tf.Variable(tf.random_normal([256, 512])),
    
                'out': tf.Variable(tf.random_normal([512, num_classes]))}
    
    biases = {  'bc1': tf.Variable(tf.random_normal([32])),
                'bc2': tf.Variable(tf.random_normal([32])),
                'bc3': tf.Variable(tf.random_normal([64])),
                'bc4': tf.Variable(tf.random_normal([128])),
                'bd1': tf.Variable(tf.random_normal([256])),
                'bd2': tf.Variable(tf.random_normal([512])),
                'out': tf.Variable(tf.random_normal([num_classes]))}
    
    trainacc = []
    costs = []
    testsacc = []
    
    
    
    
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size = 0.20)
    
    
    X = tf.placeholder(tf.float32, [None, height,width,3])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32) 

    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)
    
    # loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)
    
    
    # Evaluation
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    init = tf.global_variables_initializer()
    
    #    saver = tf.train.Saver(max_to_keep = 1)    
    #    writer = tf.summary.FileWriter('/home/alok/spyder/tensorboard/breed/1/',tf.get_default_graph())
    #    tf.summary.scalar('accuracy',accuracy)   
    #    tf.summary.scalar('cost',loss_op)  
    #    merged = tf.summary.merge_all()  
    
    cnt = 0    
    with tf.Session() as sess:
        print('session started')
        sess.run(init)
        print('initialized')
        for epoch in range(num_epochs):
            start_time = time.time()
            
            epoch_cost = 0.
            epoch_test_acc =0.
            epoch_accuracy_train = 0.
            num_minibatches = len(train_x)/minibatch_size
            
            for i in range(num_minibatches):
                dogs = []
                for j in range(minibatch_size):
                    img = cv2.imread(train_x[i*minibatch_size+j])
                    img = img/255.0
                    img = cv2.resize(img,(height,width), interpolation = cv2.INTER_CUBIC)
                    dogs.append(img)

                dogs = np.array(dogs)
                y_batch = train_y[i*minibatch_size: i*minibatch_size +minibatch_size] 
                
                _, minibatch_cost = sess.run([optimizer,loss_op],feed_dict={X:dogs, Y:y_batch, keep_prob:0.8})
#                _, minibatch_cost,summ = sess.run([optimizer,loss_op,merged],feed_dict={X:batch_x, Y:batch_y, keep_prob:0.8})

#                writer.add_summary(summ,epoch)                
                minibatch_acc = sess.run(accuracy, feed_dict={X: dogs,Y:y_batch,keep_prob: 1})

                
                test_dogs = []
                for j in range(100):

                    img = cv2.imread(test_x[cnt*50+j])
                    img = img/255.0
                    img = cv2.resize(img,(height,width), interpolation = cv2.INTER_CUBIC)
                    test_dogs.append(img)


                test_dogs = np.array(test_dogs)
                y_test = test_y[cnt*100: cnt*100 +100]
                cnt = cnt+1
                test_acc_batch = sess.run(accuracy, feed_dict={X: test_dogs,Y:y_test , keep_prob: 1})
                if(cnt == 183):
                    cnt = 0
                
                
                print(str(i)+' iter,' +' batch_cost: '+ str(minibatch_cost*0.001)+', batch_accuracy: '+str(minibatch_acc)+', test_accuray: '+str(test_acc_batch))
                
                epoch_cost += 0.001*minibatch_cost / (len(train_x)/minibatch_size)
                epoch_accuracy_train += minibatch_acc / (len(train_x)/minibatch_size)
                epoch_test_acc += test_acc_batch/ (len(train_x)/minibatch_size)
                
            costs.append(epoch_cost)
            trainacc.append(epoch_accuracy_train)
            testsacc.append(epoch_test_acc)
            pd.DataFrame(costs).to_csv('costs.csv')
            pd.DataFrame(trainacc).to_csv('trainacc.csv')
            pd.DataFrame(testsacc).to_csv('testsacc.csv')
                
            print(str(epoch)+' epoch,' +' epoch_cost: '+ str(epoch_cost)+', epoch_accuracy_train: '+str(epoch_accuracy_train)+' epoch_accuracy_test: '+ str(epoch_test_acc))                
            
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
                
                                                            
#        save after every 5th iteration
#        if(epoch %5 == 0):
#            saver.save(sess,save_path='/home/alok/spyder/checkpoints/breed/1/',global_step=epoch)
        
        plt.plot(np.arange(len(costs)),costs)
        plt.plot(np.arange(len(trainacc)),trainacc)
        plt.plot(np.arange(len(testsacc)),testsacc)
        
#        writer.close()
    return costs,trainacc,testsacc

    
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
data = pd.read_csv('all_data_labels_without_directory.csv')
names = data.id.values
breed = data.breed.values
enc = LabelEncoder()
henc = OneHotEncoder()
breed = enc.fit_transform(breed)
breed = breed.reshape(len(breed),1)
breed = henc.fit_transform(breed).toarray() 
costs,train_accuracy,tests_accuracy = main_func(names,breed,learning_rate = 0.005,num_epochs = 30,minibatch_size = 300)