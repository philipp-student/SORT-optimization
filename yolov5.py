import torch
import cv2
import numpy as np

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
img_path = './img/test_image.jpg'
img = cv2.imread(img_path)
width = img.shape[0]
height = img.shape[1]

# Inference
results = model(img_path)
results = results.xywhn[0]

num_detections = results.size(0)
detected_pedestrians = torch.zeros([num_detections, 5], dtype=torch.float)
for i in range(num_detections):
    if results[i, 5] == 0:
        detected_pedestrians[i, :] = results[i, :5]
        
num_pedestrians = detected_pedestrians.size(0)
for i in range(num_pedestrians):
    pedestrian = detected_pedestrians[i, :].numpy()    
    
    start_point = (int(pedestrian[0] * width), int(pedestrian[1] * height))
    ped_width = pedestrian[2] * width
    ped_height = pedestrian[3] * height
    end_point = ((start_point[0] + ped_width), (start_point[1] + ped_height))
    
    img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)

cv2.imshow('image', img)
cv2.waitKey(0)