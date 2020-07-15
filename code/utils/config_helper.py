# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:51:05 2020

@author: Ali Hamdi; ali.ali@rmit.edu.au
"""
 
# sacle ratio for frame resizing. 1 is means using the orignial size.
scale = 1 

# save: 0 visualises the tracking; 1 saves the tracking resutls in an excel sheet.
save = 1

# any directory to be excluded from the tracking
xdirs = []

# cnn feature extraction model
from tensorflow.keras.applications.vgg16 import VGG16
VGG16_model = VGG16(weights='imagenet', include_top=False)

# preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
preprocess = preprocess_input

#import datetime
#dt = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")