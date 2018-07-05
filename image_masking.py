#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:41:44 2018

@author: alok
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
img = cv2.imread("/home/alok/spyder/img.jpg")   

#cap = cv2.VideoCapture(0)
#while(1):
#    # Take each frame
#    _, frame = cap.read()
    # Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask= mask)
cv2.imshow('frame',frame)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
k = cv2.waitKey(5) & 0xFF
if k == 27:
    break





def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask
    
def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

def read_segmented_image(filepath, img_size):
    img = cv2.imread(os.path.join(data_dir, filepath), cv2.IMREAD_COLOR)
    img = cv2.resize(img.copy(), img_size, interpolation = cv2.INTER_AREA)

    image_mask = create_mask_for_plant(img)
    image_segmented = segment_plant(img)
    image_sharpen = sharpen_image(image_segmented)
    return img, image_mask, image_segmented, image_sharpen
     
import os
import glob
classes = []
data = []
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
for val in labels:    
    img_dir = '/home/alok/spyder/plant seedlings/train/'+labels[val] # Enter Directory of all images 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)    
    for f1 in files:


# show some images
#if show_plots:
#    for i in range(4):
# 
#        img, image_mask, image_segmented, image_sharpen = read_segmented_image(
#            train_df.loc[i,'filepath'],(224,224))
#        
#        fig, axs = plt.subplots(1, 4, figsize=(20, 20))
#        axs[0].imshow(img.astype(np.uint8))
#        axs[1].imshow(image_mask.astype(np.uint8))
#        axs[2].imshow(image_segmented.astype(np.uint8))
#        axs[3].imshow(image_sharpen.astype(np.uint8))