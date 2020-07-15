import utils.config_helper as config
import utils.bbox_helper as bbox_helper
import data.datasets as data
from collections import OrderedDict
import numpy as np
import os
import cv2
import time
import models.DroTrack as DroTrack
import math
import utils.results_saving as results_saving

s = config.scale
xdirs = config.xdirs
boxes = bbox_helper.data2bboxes(data.data)

datas = OrderedDict(sorted(data.data.items()))

for title, dataset in data.data.items():
    if title in xdirs:
        continue
    
    boxes[title] = OrderedDict(sorted(boxes[title].items()))
    for class_dir, fbbox in boxes[title].items():
        if class_dir in dataset['xdirs']:
            continue

        print (title, class_dir)

        all_boxes = bbox_helper.get_all_bboxes(dataset['url']+class_dir+'/groundtruth_rect.txt')

        for i in np.arange(len(os.listdir(dataset['url']+class_dir+'/img/'))):
            i += 1
            file = ("{0:0"+str(dataset['zc'])+"}.jpg").format(i)

            image = cv2.imread(dataset['url']+class_dir+"/img/"+file, 0)
            frame = cv2.resize(image, (0,0), fx=1/s, fy=1/s)
            #D.showImage(frame, 'frame')
            if file == (dataset['zc']-1)*'0'+'1.jpg':
                bbox = [int (fbbox[0]/s), int (fbbox[1]/s), int (fbbox[2]/s), int (fbbox[3]/s)]
                
                start = time.time()
                tracker = DroTrack.DroTrack(frame, bbox)
                end = time.time()
                extime = (end - start)
                
                gt_box = all_boxes[i-1]
                
                if config.save == 1:
                    results_saving.save_tracking_results(title, class_dir, file, bbox, bbox, s, extime)
                    
                continue    
            
            try:

                bbox, center, extime = tracker.track(frame)
    
    
                if math.isnan(all_boxes[i-1][0]) or all_boxes[i-1][0] == 'NaN':
                    gt_box = [-1,-1,0,0]
                else:
                    gt_box = all_boxes[i-1]
    
                # Visulization
                if config.save == 0:
                    bbox_helper.visualise_bbox(bbox, file, class_dir, s, dataset['url'])
                    k = cv2.waitKey(5) & 0xFF 
                    if k == 27:
                        break 
    
                if config.save == 1:
                    results_saving.save_tracking_results(title, class_dir, file, bbox, gt_box, s, extime)
            except:
                continue