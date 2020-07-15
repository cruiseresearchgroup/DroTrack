# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:01:28 2020

@author: Ali Hamdi; ali.ali@rmit.edu.au
"""

import numpy as np
import utils.bbox_helper as bbox_helper

def Angular_Relative_Scaling(angle, prev_bbox, current_center, H):
    PF_x, PF_y = bbox_helper.get_bbox_center(prev_bbox)
    F_x, F_y   = current_center
    PB_w = prev_bbox[2]
    PB_h = prev_bbox[3]
    scale = PB_h/H/5
    ratio = (F_y/PF_y)
    scaled_ratio = ratio*scale
    if angle == -90:
        F_w = PB_w-PB_w*scaled_ratio
        F_h = PB_h-PB_h*scaled_ratio
        FB = [F_x-F_w/2, F_y-F_h/2,F_w,F_h]
    elif angle < 0 and angle > -90:
        asr = scaled_ratio*np.abs(angle)/90
        F_w = PB_w-PB_w*asr
        F_h = PB_h-PB_h*asr
        FB = [F_x-F_w/2, F_y-F_h/2,F_w,F_h]
    elif angle < -90 and angle >= -180:
        asr = scaled_ratio*((180-np.abs(angle))/90)
        F_w = PB_w-PB_w*asr
        F_h = PB_h-PB_h*asr
        FB = [F_x-F_w/2, F_y-F_h/2,F_w,F_h]
    elif angle == 90:
        F_w = PB_w+PB_w*scaled_ratio
        F_h = PB_h+PB_h*scaled_ratio
        FB = [F_x-F_w/2, F_y-F_h/2,F_w,F_h]
    elif angle > 0 and angle < 90:
        asr = scaled_ratio*np.abs(angle)/90
        F_w = PB_w+PB_w*asr
        F_h = PB_h+PB_h*asr
        FB = [F_x-F_w/2, F_y-F_h/2,F_w,F_h]
    elif angle > 90 and angle <= 180:
        asr = scaled_ratio*((180-np.abs(angle))/90)
        F_w = PB_w+PB_w*asr
        F_h = PB_h+PB_h*asr
        FB = [F_x-F_w/2, F_y-F_h/2,F_w,F_h]
    else:
        # FB = [F_x, F_y, PB_w, PB_h]
        FB = prev_bbox
    return FB

def difference_direction(x, y, prev_x, prev_y):
    # compute the difference between the x and y
    # coordinates and re-initialize the direction
    # text variables

    dX = x - prev_x
    dY = y - prev_y
    (dirX, dirY) = ("", "")
    angle = np.rad2deg(np.arctan2(dY, dX))

    # ensure there is significant movement in the
    # x-direction
    if np.abs(dX) > 0:
        dirX = "East" if np.sign(dX) == 1 else "West"

    # ensure there is significant movement in the
    # y-direction
    if np.abs(dY) > 0:
        dirY = "South" if np.sign(dY) == 1 else "North"

    # handle when both directions are non-empty
    if dirX != "" and dirY != "":
        direction = "{}-{}".format(dirY, dirX)

    # otherwise, only one direction is non-empty
    else:
        direction = dirX if dirX != "" else dirY

    return dX, dY, direction, angle



# Angular scaling
def out_of_view_correction(frame, center, prev_of_point):
    ofx = ''
    ofy = ''
    
    if center[0] > frame.shape[1]:
        ofx = frame.shape[1] * .52
    if center[0] < 0:
        ofx = frame.shape[1] * .48
    
    if center[1] > frame.shape[0]:
        ofy = frame.shape[0] * .52
    if center[1] < 0:
        ofy = frame.shape[0] * .48
    #
    if ofx != '' and ofy != '':    
        center = (int(ofx), int(ofy))
    elif ofx != '' and ofy == '':
        center = (int(ofx), center[1])
    elif ofx == '' and ofy != '':
        center = (center[0], int(ofy))
        
    final_dx, final_dy, final_direction, final_angle = difference_direction(center[0], center[1], prev_of_point[0], prev_of_point[1])
    
    return center, final_angle

def angular_bbox_correction(bbox, frame, fbbox):
    if bbox[0] < 0:
        bbox[0] == -bbox[0]*4
    if bbox[1] < 0:
        bbox[1] == -bbox[1]*4
    
    if bbox[0] > frame.shape[1]:
        bbox[0] == frame.shape[1]
    if bbox[1] > frame.shape[0]:
        bbox[1] == frame.shape[0]
        
    if bbox[2] > 1.05*fbbox[2]:
        bbox[2] = fbbox[2]*1.05
    if bbox[2] < .95*fbbox[2]:
        bbox[2] = fbbox[2]*.95

    if bbox[3] > 1.05*fbbox[3]:
        bbox[3] = fbbox[3]*1.05
    if bbox[3] < .95*fbbox[3]:
        bbox[3] = fbbox[3]*.95
    return bbox
