import os
import numpy as np
from sort import associate_detections_to_trackers
from tracking import draw_bounding_boxes
from PIL import Image
import cv2
from helpers import xywh2xyxy

# Main folder with all datasets.
MAIN_FOLDER = r"D:\Philipp Student\HRW\Fahrassistenzsysteme 2\Seminararbeit\MOT15\train"

for dataset_folder in os.listdir(MAIN_FOLDER):
    
    # Dataset main folder.
    DATASET_MAIN_FOLDER = os.path.join(MAIN_FOLDER, dataset_folder)

    # Name of dataset.
    dataset_name = dataset_folder

    # Images of dataset.
    DATASET_IMAGES_FOLDER = os.path.join(DATASET_MAIN_FOLDER, "img1")

    # Groundtruth file (custom format).
    groundtruth_file = os.path.join(DATASET_MAIN_FOLDER, "gt", "gt_custom.txt")

    # Result file.
    #result_file = os.path.join(DATASET_MAIN_FOLDER, "tracks", "tracks.txt")                        # Tracks
    result_file = os.path.join(DATASET_MAIN_FOLDER, "det", "det_custom.txt")                       # Original Detections
    #result_file = os.path.join(DATASET_MAIN_FOLDER, "det", "YOLOv5_" + dataset_name + ".txt")      # YOLOv5 detections
    
    ############ Load data ############
    
    groundtruths = np.loadtxt(groundtruth_file, delimiter=',')
    results = np.loadtxt(result_file, delimiter=',')
    
    # Get number of frames in dataset.
    num_frames = int(np.max(groundtruths[:, 0]))
    
    for frame_index in range(1, num_frames):
        
        # Construct path to current image.
        IMAGE_PATH = os.path.join(DATASET_IMAGES_FOLDER, format(frame_index, '06d') + ".jpg")
        
        # Load current image.
        IMAGE = np.array(Image.open(IMAGE_PATH))
        
        # Get all groundtruth bounding boxes for current frame.
        groundtruth_boxes = groundtruths[groundtruths[:, 0] == frame_index, 1:]
        
        # Get all result bounding boxes for current frame.
        result_boxes = results[results[:, 0] == frame_index, 1:]
        
        # Display groundtruths and results.
        IMAGE = draw_bounding_boxes(IMAGE, groundtruth_boxes, (255, 0, 0), False)
        IMAGE = draw_bounding_boxes(IMAGE, result_boxes, (0, 255, 0), False)
        
        # Display image.    
        cv2.imshow('image', IMAGE)
        cv2.waitKey(1)                
    
        # Compute correspondences of bounding boxes.
        #matches, unmatched_groundtruths, unmatched_results = associate_detections_to_trackers(groundtruth_boxes, result_boxes, 'iou', 0.3)
    
        pass