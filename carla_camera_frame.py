from PIL import Image

class CARLA_Camera_Frame:
    
    def __init__(self, frame, frame_index):
        self.frame = frame
        self.frame_index = frame_index
        
    def save_frame(self, path):
        Image.fromarray(self.frame).save(path)