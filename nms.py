# -*- coding: utf-8 -*-
import numpy as np

detections = np.loadtxt("detections.txt")
detections = detections.reshape(-1,5)

def nms(detections, threshold):
    x1 = detections[:,0]
    y1 = detections[:,1]
    x2 = detections[:,2]
    y2 = detections[:,3]
    scores = detections[:,4]
    
    #compute each box area
    area = (x2-x1+1)*(y2-y1+1)
    #order by scores
    index = scores.argsort()[::-1]
    
    keep = []
    while index.size > 0:
        i = index[0]

        keep.append(i)
        #compute iou
        xx1 = np.maximum(x1[i], x1[index[1:]])
        yy1 = np.maximum(y1[i], y1[index[1:]])
        xx2 = np.minimum(x2[i], x2[index[1:]])
        yy2 = np.minimum(y2[i], y2[index[1:]])
        
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        
        inter = w*h
        
        iou = inter/(area[i]+area[index[1:]]-inter)
        index_keep = np.where(iou<=threshold)[0]
        index = index[index_keep+1]
        
    return keep

res = nms(detections, 0.3)
