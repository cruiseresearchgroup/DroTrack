# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:37:39 2020

@author: Ali Hamdi; ali.ali@rmit.edu.au
"""

from collections import OrderedDict
import cv2
import numpy as np
from scipy.spatial import distance

def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou


def data2bboxes(data):
    boxes = {}
    for title, dataset in data.items():
        boxes.update({title:{}})
        for class_dir in dataset['dirs']:
            with open(dataset['url']+class_dir+'/groundtruth_rect.txt') as f:
                first_line = f.readline()
            points = []
            try:
                for point in first_line.split(','):
                    points.append(int(point))
            except:
                for point in first_line.split('\t'):
                    points.append(int(point))
            bbox = tuple(points)
            boxes[title][class_dir] = bbox
        return OrderedDict(sorted(boxes.items()))

def get_all_bboxes(filename):    
    f = open(filename, 'r')
    lines = f.readlines()
    
    all_boxes = []
    for line in lines:
        box = []
        for point in line.split(','):
            try:
                box.append(float(point))
            except:
                box.append(np.nan)
        all_boxes.append(box)
    return all_boxes

def get_bbox_points(bbox):
    y1 = int(bbox[1])
    y2 = int(bbox[1])+int(bbox[3])
    x1 = int(bbox[0])
    x2 = int(bbox[0])+int(bbox[2])
    return x1, y1, x2, y2

def visualise_bbox(bbox, file, class_dir, s, dataset):
    frame_color  = cv2.imread(dataset+class_dir+"/img/"+file)
    x1, y1, x2, y2 = get_bbox_points(bbox)
    x1, y1, x2, y2 = x1*s, y1*s, x2*s, y2*s
    cv2.rectangle(frame_color, (x1,y1), (x2,y2), (0,255,255), 1)
    cv2.imshow(class_dir, frame_color)
    
def get_bbox_center(bbox):
    bbox_center_x = int(bbox[0]+(bbox[2]/2))
    bbox_center_y = int(bbox[1]+(bbox[3]/2))
    bc_point = (bbox_center_x, bbox_center_y)
    return bc_point
    
# Calculate the difference between two points; ex: orginal center point and tracked point
def complement_point(point, ppoint):
    return point[0]-ppoint[0], point[1]-ppoint[1]