#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 17:09:52 2018

@author: alok
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


    
#img = cv2.imread('0a0c223352985ec154fd604d7ddceabd.jpg')
#imgg = cv2.resize(img,(60,60), interpolation = cv2.INTER_CUBIC)
#plt.imshow(img)

#rows,cols,depth = img.shape

def rotate_image(img,angle = 90):
    rows,cols,depth = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    rot = cv2.warpAffine(img,M,(cols,rows))    
    return rot

def shear_image(img):
    rows,cols,depth = img.shape
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def perspective_image(img):
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(300,300))
    return dst

def average_image(img):
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    return dst

def blur_image(img):
    blur = cv2.bilateralFilter(img,9,75,75)
    return blur

def crop_image(img,x = 10,y = 10):
    row = img.shape[1] - x
    col = img.shape[0] - y
    crop_img = img[x:row,y:col]
    return crop_img

def flip_horizontal(img):
    return cv2.flip( img, 0 )  
  
def flip_hvertical(img):
    return cv2.flip( img, 1)


data = pd.read_csv('labels.csv')    
#import time
#from datetime import timedelta
labels = []
breed = []  
nasals = data.breed.value_counts().index.values
for i in range(len(nasals)):       
    filenames = data[data.breed == nasals[i]].id.values
#    start_time = time.time()   
    for j in range(len(filenames)):
       
            
#        img = cv2.imread(filenames[j]+'.jpg')
#        path = '/home/alok/spyder/dogbreed/augumented/' +str(i)+'/'+ filenames[j]+'_'+str(0)+'.jpg'
        labels.append(str(i)+'/'+ filenames[j]+'_'+str(0)+'.jpg')
        breed.append(nasals[i])
#        cv2.imwrite(path, img)
#        
#        img1 = rotate_image(img)
#        path = '/home/alok/spyder/dogbreed/augumented/' +str(i)+'/'+ filenames[j]+'_'+str(1)+'.jpg'
        labels.append(str(i)+'/'+ filenames[j]+'_'+str(1)+'.jpg')
        breed.append(nasals[i])
#        cv2.imwrite(path, img1)
#        
#        img2 = perspective_image(img)
#        path = '/home/alok/spyder/dogbreed/augumented/' +str(i)+'/'+ filenames[j]+'_'+str(2)+'.jpg'
        labels.append(str(i)+'/'+ filenames[j]+'_'+str(2)+'.jpg')
        breed.append(nasals[i])
#        cv2.imwrite(path, img2)
#        
#        img3 = shear_image(img)
#        path = '/home/alok/spyder/dogbreed/augumented/' +str(i)+'/'+ filenames[j]+'_'+str(3)+'.jpg'
        labels.append(str(i)+'/'+ filenames[j]+'_'+str(3)+'.jpg')
        breed.append(nasals[i])
#        cv2.imwrite(path, img3)
#        
#        img4 = average_image(img)
#        path = '/home/alok/spyder/dogbreed/augumented/' +str(i)+'/'+ filenames[j]+'_'+str(4)+'.jpg'
        labels.append(str(i)+'/'+ filenames[j]+'_'+str(4)+'.jpg')
        breed.append(nasals[i])
#        cv2.imwrite(path, img4)
#        
#        img5 = blur_image(img)
#        path = '/home/alok/spyder/dogbreed/augumented/' +str(i)+'/'+ filenames[j]+'_'+str(5)+'.jpg'
        labels.append(str(i)+'/'+ filenames[j]+'_'+str(5)+'.jpg')
        breed.append(nasals[i])
#        cv2.imwrite(path, img5)
#        
#        img6 = crop_image(img)
#        path = '/home/alok/spyder/dogbreed/augumented/' +str(i)+'/'+ filenames[j]+'_'+str(6)+'.jpg'
        labels.append(str(i)+'/'+ filenames[j]+'_'+str(6)+'.jpg')
        breed.append(nasals[i])
#        cv2.imwrite(path, img6)
#        
#        img7 = flip_horizontal(img)
#        path = '/home/alok/spyder/dogbreed/augumented/' +str(i)+'/'+ filenames[j]+'_'+str(7)+'.jpg'
        labels.append(str(i)+'/'+ filenames[j]+'_'+str(7)+'.jpg')
        breed.append(nasals[i])
#        cv2.imwrite(path, img7)
#        
#        img8 =  flip_hvertical(img)
#        path = '/home/alok/spyder/dogbreed/augumented/' +str(i)+'/'+ filenames[j]+'_'+str(8)+'.jpg'
        labels.append(str(i)+'/'+ filenames[j]+'_'+str(8)+'.jpg')
        breed.append(nasals[i])
#        cv2.imwrite(path, img8)
        
#    end_time = time.time()
#    time_dif = end_time - start_time
#    print(str(i)+" Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        
labels = np.asarray(labels) 
breed =  np.asarray(breed)   
labels = pd.DataFrame({'id':labels,'breed':breed})
labels.to_csv('all_data_labels.csv')
    
    
    
#for i in range(120):    
#    newpath = '/home/alok/spyder/dogbreed/augumented/' +str(i)
#    if not os.path.exists(newpath):
#        os.makedirs(newpath)






