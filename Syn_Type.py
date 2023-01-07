# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 18:37:30 2022

@author: Administrator
"""
import cv2
import numpy as np
import random
import math
import torch
import numpy as np
import cv2
import time
import os
from LPSLE_Model import *
import utils_train

# classification
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt



def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])

def LowLight(img):

    g = np.random.uniform(0.1,0.2)
    img_l = img*g
    
    return img_l
    

def Hazey(img):
    a = np.random.uniform(0.70,0.95)       
    t = np.random.uniform(0.1,0.5)
    img_l = img*t +a*(1-t)
    return img_l
# epoch = 1
# test_dir1 = './dataset/Test_Lowlight'
# testfiles1 = os.listdir(test_dir1)
# result_dir = './result'
# for f in range(len(testfiles1)):
#     img = cv2.imread(test_dir1 + '/' + testfiles1[f])/255
# # img = cv2.imread('2.jpg')/255
# #     cv2.imwrite('output1.jpg',Hazey(img)*255)
#     cv2.imwrite(result_dir + '/' + testfiles1[f][:-4] + '_%d_9' % (epoch) + '.png',
#                 Hazey(img)*255)
#
img = cv2.imread('4.jpg')/255
cv2.imwrite('output4.jpg',Hazey(img)*255)
