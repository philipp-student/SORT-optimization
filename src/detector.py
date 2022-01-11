import torch

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

        # Get number of total detections.
        num_detections = results.size(0)
        
        # Get coordinates of all detected pedestrians.
        detected_pedestrians = torch.zeros([num_detections, 5], dtype=torch.float)
        for i in range(num_detections):
            if results[i, 5] == 0:
                detected_pedestrians[i, :] = results[i, :5]
                
        # Return detected pedestrians.
        return detected_pedestrians
        