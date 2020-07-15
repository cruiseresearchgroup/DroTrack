# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 21:17:11 2020

@author: Ali Hamdi; ali.ali@rmit.edu.au
"""

import utils.bbox_helper as bbox_helper
import utils.config_helper as config
from scipy.spatial import distance
import pandas as pd

def save_tracking_results(title, class_dir, file, bbox, gt_box, s, extime):
    center   = bbox_helper.get_bbox_center(bbox)
    gtcenter = bbox_helper.get_bbox_center(gt_box)

    results = [title, class_dir, file, bbox[0]*s, bbox[1]*s, bbox[2]*s, bbox[3]*s, 
                                   center[0]*s, center[1]*s, 
                                   bbox_helper.intersection_over_union(bbox_helper.get_bbox_points(gt_box),
                                                              tuple(i * s for i in bbox_helper.get_bbox_points(bbox))),
                                   distance.euclidean((gtcenter[0], gtcenter[1]), (center[0]*s, center[1]*s)),
                                   extime]
    dataframe = pd.DataFrame([results], columns=None)
    with open('results/DroTrack_results_evaluation.csv', 'a') as f:
        dataframe.to_csv(f, index = False, header=None)