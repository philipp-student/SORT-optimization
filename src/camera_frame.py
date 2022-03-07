from PIL import Image
import numpy as np

class CameraFrame:

    def __init__(self, frame, frame_index):

        self.frame = frame

        self.frame_index = frame_index
        

    def save_frame(self, path):
        Image.fromarray(self.frame).save(path)
    
    @staticmethod    
    def from_image(image_path, frame_index):
        # Read image and convert to numpy.
        img = np.array(Image.open(image_path))
        
        # Create frame and return it.
        return CameraFrame(img, frame_index)