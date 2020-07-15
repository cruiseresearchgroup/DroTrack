# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:44:57 2020

@author: Ali Hamdi; ali.ali@rmit.edu.au
"""

import cv2
import numpy as np

def features(image, model, preprocess_input):
    s = 32 #max (image.shape[0], image.shape[0], 32)
    image = cv2.resize(image, (s,s))
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    #print (image.shape)
    x = np.expand_dims(image, axis=0)

    #x = np.array(x, dtype=np.float64)
    x = preprocess_input(x)#.astype('float32')

    features = model.predict(x)
    #print (features.shape)
    features = features.flatten()
    #del model, x
    #gc.collect()
        
    return features