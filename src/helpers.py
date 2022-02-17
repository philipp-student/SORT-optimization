import numpy as np

def xywh2xyxy(bbox_xywh):
    x_center = bbox_xywh[:, 0]
    y_center = bbox_xywh[:, 1]
    width = bbox_xywh[:, 2]
    height = bbox_xywh[:, 3]
    
    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)

    x2 = x1 + width
    y2 = y1 + height
    
    return np.array([x1, y1, x2, y2])