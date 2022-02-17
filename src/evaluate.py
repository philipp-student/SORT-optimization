import os
import numpy as np
from sort import associate_detections_to_trackers
from tracking import draw_bounding_boxes
from PIL import Image
import cv2
from helpers import xywh2xyxy

def recall(matches, unmatched_groundtruths, unmatched_results):
    tp = matches.shape[0]
    fp = unmatched_results.shape[0]
    fn = unmatched_groundtruths.shape[0]
    
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0

def precision(matches, unmatched_groundtruths, unmatched_results):
    tp = matches.shape[0]
    fp = unmatched_results.shape[0]
    fn = unmatched_groundtruths.shape[0]
    
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0
        
# Main folder with all datasets.
MAIN_FOLDER = r"D:\Philipp Student\HRW\Fahrassistenzsysteme 2\Seminararbeit\MOT15\train"

for dataset_folder in os.listdir(MAIN_FOLDER):
    
    # Dataset main folder.
    DATASET_MAIN_FOLDER = os.path.join(MAIN_FOLDER, dataset_folder)
    
    # Detections folder.
    DETECTIONS_FOLDER = os.path.join(DATASET_MAIN_FOLDER, "det")
    
    # Groundtruths folder.
    GROUNDTRUTHS_FOLDER = os.path.join(DATASET_MAIN_FOLDER, "gt")
    
    # Groundtruths folder.
    TRACKS_FOLDER = os.path.join(DATASET_MAIN_FOLDER, "tracks")

    # Name of dataset.
    dataset_name = dataset_folder
    print("################# {0} #################".format(dataset_name))

    # Groundtruth file (custom format).
    groundtruth_file = os.path.join(GROUNDTRUTHS_FOLDER, "gt_custom.txt")                   # Groundtruths.

    # Result file.
    #result_file = os.path.join(TRACKS_FOLDER, "tracks.txt")                                # Tracks
    #result_file = os.path.join(DETECTIONS_FOLDER, "det_custom.txt")                        # Original Detections
    result_file = os.path.join(DETECTIONS_FOLDER, "det_YOLOv5.txt")                         # YOLOv5 detections
    
    ############ Load data ############
    
    groundtruths = np.loadtxt(groundtruth_file, delimiter=',')
    results = np.loadtxt(result_file, delimiter=',')
    
    # Get number of frames in dataset.
    num_frames = int(np.max(groundtruths[:, 0]))
    
    precision_sum = 0
    recall_sum = 0
    
    for frame_index in range(1, num_frames):
        
        # Get all groundtruth bounding boxes for current frame.
        groundtruth_boxes = groundtruths[groundtruths[:, 0] == frame_index, 1:]
        
        # Get all result bounding boxes for current frame.
        result_boxes = results[results[:, 0] == frame_index, 1:]
    
        # Compute correspondences of bounding boxes.
        matches, unmatched_groundtruths, unmatched_results = associate_detections_to_trackers(groundtruth_boxes, result_boxes[:, :4], 'iou', 0.5)
    
        # Compute evaluation metrics.
        image_precision = precision(matches, unmatched_groundtruths, unmatched_results)
        image_recall = recall(matches, unmatched_groundtruths, unmatched_results)
        
        # Cumulate evaluation metrics to compute an average.
        precision_sum += image_precision
        recall_sum += image_recall

    # Print evaluation metrics.
    print("Average precision: {0}".format(precision_sum / num_frames))
    print("Average recall: {0}".format(recall_sum / num_frames))

    pass    