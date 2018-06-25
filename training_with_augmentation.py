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

def conv2d(x, W, b, strides=1,padding = 'VALID'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x,f = 2, k=2,padding = 'VALID'):
    return tf.nn.max_pool(x, ksize=[1, f, f, 1], strides=[1, k, k, 1],padding='VALID')


def conv_net(x, weights, biases, dropout):

    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
#    x = tf.reshape(x, shape=[-1, 80, 80, 3])
#    print(x.shape)
    conv1 = conv2d(x, weights['wc1'], biases['bc1'],strides=2)
#    print(conv1.shape)
#    conv1 = maxpool2d(conv1,f=2, k=2)
#    print(conv1.shape)
    conv1 = tf.nn.relu(conv1)
#    print(conv1.shape)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'],strides=1)
#    print(conv2.shape)
#    conv2 = maxpool2d(conv2, k=2)
#    print(conv2.shape)
    conv2 = tf.nn.relu(conv2)
#    print(conv2.shape)


    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'],strides=1,padding='SAME')
#    print(conv3.shape)
    conv3 = maxpool2d(conv3,f=3, k=2)
#    print(conv3.shape)
    conv3 = tf.nn.relu(conv3)
#    print(conv3.shape)

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'],strides=1)
    print(conv4.shape)
#    conv4 = maxpool2d(conv4,f=3, k=2)
#    print(conv3.shape)
    conv4 = tf.nn.relu(conv4)



#    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'],strides=2)
#    print(conv3.shape)
#    conv5 = maxpool2d(conv5,f=3, k=2)
#    print(conv3.shape)
#    conv5 = tf.nn.relu(conv5)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
#    print(fc1.shape)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
#    print(fc1.shape)
    fc1 = tf.nn.relu(fc1)
#    print(fc1.shape)
    #Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    
    
    

#    print(fc1.shape)
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
#    print(fc1.shape)
    fc2 = tf.nn.relu(fc2)
#    print(fc1.shape)
    #Dropout
    fc2 = tf.nn.dropout(fc2, dropout)
    # Output, class prediction
    
    
    
    
    
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

#def main_func():
        # Parameters
num_classes = 120 

weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([71*71*256, 2048])),
    'wd2': tf.Variable(tf.random_normal([2048, 1024])),

    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([2048])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

learning_rate = 0.005
num_epochs = 1
num_classes = 120 
#minibatch_size = 50
acc = []
costs = []
#    num_input = 10000#784 
#dropout = 0.75 

X = tf.placeholder(tf.float32, [None, 299,299,3])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
#  model
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
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
data = pd.read_csv('all_data_labels_without_directory.csv')
names = data.id.values
breed = data.breed.values
enc = LabelEncoder()
henc = OneHotEncoder()
breed = enc.fit_transform(breed)
breed = breed.reshape(len(breed),1)
breed = henc.fit_transform(breed).toarray()


    
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        epoch_cost = 0.
        epoch_accuracy = 0.
#        num_minibatches = int(10222 / minibatch_size)
#        minibatches = random_mini_batches(dogs, breed, minibatch_size)
        
        for i in range(1839):
            dogs = []
            for j in range(50):
                img = cv2.imread(names[i*50+j])
                img = img/255.0
                img = cv2.resize(img,(299,299), interpolation = cv2.INTER_CUBIC)
                dogs.append(img)
                print(i,j)
        #        if(i*50 + j == 91997):
        #            break
            dogs = np.array(dogs)
            y = breed[i*50: i*50 +50] 
            print('fifty images')
    
#        for minibatch in minibatches:
#            
#            (batch_x, batch_y) = minibatch
##                print(batch_x.shape, batch_y.shape)
#
            _, minibatch_cost = sess.run([optimizer,loss_op],feed_dict={X:dogs, Y:y, keep_prob:0.8})
#            _, minibatch_cost = sess.run([optimizer,loss_op],feed_dict={X:batch_x, Y:batch_y, keep_prob:0.8})
##                _, minibatch_cost,summ = sess.run([optimizer,loss_op,merged],feed_dict={X:batch_x, Y:batch_y, keep_prob:0.8})
            print('gradient complete')
            epoch_cost += 0.001*minibatch_cost / 1839
#  
##            writer.add_summary(summ,epoch)                
            minibatch_acc = sess.run(accuracy, feed_dict={X: dogs,Y:y,keep_prob: 1.0})
            print('accuracy complete')
            epoch_accuracy += minibatch_acc / 1839
            
            
        costs.append(epoch_cost)
        acc.append(epoch_accuracy)
            
        print(str(epoch)+' epoch,' +' epoch_cost: '+ str(epoch_cost)+', epoch_accuracy: '+str(epoch_accuracy))         
#        print ("Train Accuracy batch:", sess.run([accuracy], feed_dict={X: batch_x,Y:batch_y,keep_prob: 1.0}))
#            print ("Train Accuracy:", sess.run([accuracy], feed_dict={X: dogs[:5000],Y:breed[0:5000],keep_prob: 1.0}))
#            print ("Test Accuracy:", sess.run([accuracy], feed_dict={X: mnist.test.images,Y:mnist.test.labels,keep_prob: 1.0}))
        
        
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            
        
#        save after every 5th iteration
#            if(epoch %5 == 0):
#                saver.save(sess,save_path='/home/alok/spyder/checkpoints/breed/1/',global_step=epoch)
    
#    plt.plot(np.arange(len(costs)),costs)
#    plt.plot(np.arange(len(acc)),acc)
    
#    writer.close()
#    return costs,acc
#
#costs,acc= main_func()
     
