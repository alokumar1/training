#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:29:29 2018

@author: alok
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 18:05:30 2018

@author: alok
"""

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
  
def flip_vertical(img):
    return cv2.flip( img, 1)

def get_augment(data,directory_to_save):
    image_name = []
    breed = []
    for i in range(len(data)):
        random = np.random.permutation(7)
        img_raw = cv2.imread(data.id[i]+'.jpg')  
        
        for item in random[0:4]:
            path = directory_to_save + data.id[i]+'_'+str(item)+'.jpg'
            img = aug_dict[item](img_raw)
            cv2.imwrite(path, img)
            image_name.append(data.id[i] +'_'+str(item)+'.jpg')
            breed.append(data.breed[i])
            
        cv2.imwrite(directory_to_save+ data.id[i]+'.jpg', img_raw)
        image_name.append(data.id[i]+'.jpg')
        breed.append(data.breed[i])     
        
    labels = pd.DataFrame({'id':image_name,'breed':breed})       
    
    return labels
    
aug_dict = {0:rotate_image,
            1:shear_image,
            2:average_image,
            3:blur_image,
            4:crop_image,
            5:flip_vertical,
            6:perspective_image}

data = pd.read_csv('labels.csv')  
labels = get_augment(data,directory_to_save= '/home/alok/spyder/dogbreed/five_times_images/')
labels.to_csv('five_times_images_data.csv')