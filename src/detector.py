import torch
import numpy as np

PERSON_DETECTION_ID = 0

class Detector:
    def __init__(self):
        self.model = None
        
    def detect_pedestrians(self):
        pass

class YOLOv5(Detector):
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = torch.hub.load('ultralytics/yolov5', self.model_type, pretrained=True)
        
    def detect_pedestrians(self, img):
        # Inference.
        results = self.model(img)
        
        # Get object bounding box info from results.
        results = results.xyxyn[0]
        
        # Get rows that contain pedestrian detections.
        pedestrian_rows = np.where(results[:,5] == PERSON_DETECTION_ID)

        # Select rows with pedestrians.
        results = results[pedestrian_rows]
        
        # Get number of detected pedestrians.
        num_detections = results.size(0)

        # Get coordinates of all detected pedestrians.
        detected_pedestrians = torch.zeros([num_detections, 4], dtype=torch.float)
        for i in range(num_detections):
            detected_pedestrians[i, :] = results[i, :4]
                
        # Return detected pedestrians.
        return detected_pedestrians
        