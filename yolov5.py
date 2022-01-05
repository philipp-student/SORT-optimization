import torch
import cv2
import numpy as np

def xywh2xyxy(xywh, width, height):
    center = (xywh[0] * width, xywh[1] * height)
    
    obj_width = xywh[2] * width
    obj_height = xywh[3] * height
    
    start_point = (center[0] - (obj_width/2), center[1] - obj_height/2)    
    end_point = (start_point[0] + obj_width, start_point[1] + obj_height)
    
    return np.array([start_point[0], start_point[1], end_point[0], end_point[1]])

def norm2img(xyxy, width, height):
    return np.array([xyxy[0] * width, xyxy[1] * height, xyxy[2] * width, xyxy[3] * height])

def plot_box(img, pedestrian_xyxy, width, height, color=(255, 0, 0), thickness=2):        
    # Compute actual image coordinates of starting and end points.
    pedestrian_xyxy = norm2img(pedestrian_xyxy, width, height)
    
    # Plot rectangle in image.
    return cv2.rectangle(img, (int(pedestrian_xyxy[0]), int(pedestrian_xyxy[1])), (int(pedestrian_xyxy[2]), int(pedestrian_xyxy[3])), color, thickness)

# Load model.
model_type = 'yolov5s'
model = torch.hub.load('ultralytics/yolov5', model_type, pretrained=True)

# Load images.
img_path = './img/2.jpg'
img = cv2.imread(img_path)
width = img.shape[1]
height = img.shape[0]

# Inference.
results = model(img_path)

# Get object bounding box info from results.
results = results.xyxyn[0]

# Get number of total detections.
num_detections = results.size(0)

if (num_detections != 0):    
    # Get coordinates of all detected pedestrians.
    detected_pedestrians = torch.zeros([num_detections, 5], dtype=torch.float)
    for i in range(num_detections):
        if results[i, 5] == 0:
            detected_pedestrians[i, :] = results[i, :5]

    # Draw bounding box around each detected pedestrian.        
    num_pedestrians = detected_pedestrians.size(0)
    for i in range(num_pedestrians):
        # Get coordinates of bounding box.
        pedestrian = detected_pedestrians[i, :].numpy()    
        
        # Plot bounding box.
        img = plot_box(img, pedestrian, width, height)

    # Display image.
    cv2.imshow('image', img)
    cv2.waitKey(0)