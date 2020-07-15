# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:13:49 2020

@author: Ali Hamdi; ali.ali@rmit.edu.au
"""

import cv2
import utils.bbox_helper as bbox_helper
import numpy as np

def find_Shi_Tomasi_corners(first_frame, maxCorners = 100, qualityLevel = 0.2, minDistance = 7, blockSize = 7):
    # ShiTomasi corner detection
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
    # Set parameters for 
    feature_params = dict(maxCorners=maxCorners, 
                          qualityLevel=qualityLevel, 
                          minDistance=minDistance, 
                          blockSize=blockSize) #,useHarrisDetector = 1
    # Find inital corner locations
    corners = cv2.goodFeaturesToTrack(first_frame, mask = None, **feature_params)
    return corners

def LKT_intialization(first_frame, template, bbox, 
                      maxCorners = 100, 
                      qualityLevel = 0.5, 
                      minDistance = 1, 
                      blockSize = 3, padding = 0):
    
    qualityLevel = max(qualityLevel, 0.005)
    
    # ShiTomasi corner detection 
    first_frame_corners = find_Shi_Tomasi_corners(first_frame, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance, blockSize=blockSize)

    corners = []
    for i in first_frame_corners:
        x,y = i.ravel()
        corners.append(list(i.ravel()))

    points = []
    x1, y1, x2, y2 = bbox_helper.get_bbox_points(bbox)

    for i in range(len(corners)):

        p_x = corners[i][0] #Current point x
        p_y = corners[i][1] #Current point y

        if ((p_x >= x1 and p_x < x2) and (p_y >= y1 and p_y < y2)):
            points.append(corners[i])

    if len(points) > 0:
        return np.array(points)
    else:
        maxCorners = maxCorners + 10; 
        qualityLevel = qualityLevel - 0.05; 
        minDistance = minDistance+1; 
        blockSize = blockSize+1; 
        padding = padding+1
        return LKT_intialization(first_frame, template, bbox, 
                                 maxCorners, 
                                 qualityLevel, 
                                 minDistance, 
                                 blockSize, 
                                 padding)
        
def otpical_flow_LKT(prev_frame, frame, prev_corners, mask, scale):

    s = int(100*scale)
    if s < 15:
        s = 15
    if s > 30:
        s = 30

    m = int(s/2)*10
        
    # Set parameters for lucas kanade optical flow
    lucas_kanade_params = dict( winSize  = (s, s), # (int(500*scale), int(500*scale)), 
                                maxLevel = 10,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, m, m/100)
                               )
    # calculate optical flow
    current_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_frame, frame,  prev_corners,  None, **lucas_kanade_params)
    
    return current_corners, status, errors