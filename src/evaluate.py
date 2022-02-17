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

MODE = "detection"
DETECTOR_TYPE = "original"

# Main folder with all datasets.
MAIN_FOLDER = r"D:\Philipp Student\HRW\Fahrassistenzsysteme 2\Seminararbeit\MOT15\train"

for dataset_folder in os.listdir(MAIN_FOLDER):
    
    ############ Determine paths ############
    
    # Dataset main folder.
    DATASET_MAIN_FOLDER = os.path.join(MAIN_FOLDER, dataset_folder)
    
    # Detections folder.
    DETECTIONS_FOLDER = os.path.join(DATASET_MAIN_FOLDER, "det")
    
    # Groundtruths folder.
    GROUNDTRUTHS_FOLDER = os.path.join(DATASET_MAIN_FOLDER, "gt")
    
    # Tracks folder.
    TRACKS_FOLDER = os.path.join(DATASET_MAIN_FOLDER, "tracks")
    
    # Evaluation folder.
    EVALUATION_FOLDER = os.path.join(DATASET_MAIN_FOLDER, "eval")

    # Name of dataset.
    dataset_name = dataset_folder
    print("################# {0} #################".format(dataset_name))

    # Groundtruth file (custom format).
    groundtruths_file = os.path.join(GROUNDTRUTHS_FOLDER, "gt_custom.txt")                           # Groundtruths

    # Orignal detections file.
    original_detections_file = os.path.join(DETECTIONS_FOLDER, "det_custom.txt")                    # Original Detections

    # YOLOv5 detections file.
    yolo_detections_file = os.path.join(DETECTIONS_FOLDER, "det_YOLOv5.txt")          # YOLOv5 detections

    # Tracks file.
    tracks_file = os.path.join(TRACKS_FOLDER, "tracks.txt")                                         # Tracks
    
    # Evaluation file.
    if MODE == 'detection':
        if DETECTOR_TYPE == 'original':
            evaluation_file = os.path.join(EVALUATION_FOLDER, "evaluation_original_detector.txt")
        elif DETECTOR_TYPE == 'yolov5':
            evaluation_file = os.path.join(EVALUATION_FOLDER, "evaluation_yolov5_detector.txt")
    elif MODE == 'tracking':
            evaluation_file = os.path.join(EVALUATION_FOLDER, "evaluation_tracking.txt")
    
    # Create evaluation folder if it does not exist yet.
    if not os.path.exists(EVALUATION_FOLDER):
        os.mkdir(EVALUATION_FOLDER)
    
    ############ Load data ############
    
    groundtruths = np.loadtxt(groundtruths_file, delimiter=',')
    original_detections = np.loadtxt(original_detections_file, delimiter=',')
    yolo_detections = np.loadtxt(yolo_detections_file, delimiter=',')
    #tracks = np.loadtxt(tracks_file, delimiter=',')
    
    ############ Evaluate ############
    
    # Get number of frames in dataset.
    num_frames = int(np.max(groundtruths[:, 0]))
    
    if MODE == 'detection':
        evaluation_values = np.zeros((num_frames, 3))
    elif MODE == 'tracking':
        # TODO: Implement!
        pass
    
    precision_sum = 0
    recall_sum = 0
    
    for frame_index in range(1, num_frames + 1):

        # Get all groundtruth bounding boxes for current frame.
        groundtruth_boxes = groundtruths[groundtruths[:, 0] == frame_index, 1:]
        
        if MODE == 'detection':
            if DETECTOR_TYPE == 'original':
                # Get all original detection bounding boxes for current frame.
                result_boxes = original_detections[original_detections[:, 0] == frame_index, 1:]
            elif DETECTOR_TYPE == 'yolov5':
                # Get all YOLOv5 detection bounding boxes for current frame.
                result_boxes = yolo_detections[yolo_detections[:, 0] == frame_index, 1:]
                
            # Compute correspondences of bounding boxes.
            matches, unmatched_groundtruths, unmatched_results = associate_detections_to_trackers(groundtruth_boxes, result_boxes[:, :4], 'iou', 0.5)
            
            # Compute evaluation metrics.
            image_precision = precision(matches, unmatched_groundtruths, unmatched_results)
            image_recall = recall(matches, unmatched_groundtruths, unmatched_results)
            
            # Save evaluation values.
            evaluation_values[frame_index - 1, :] = np.array([frame_index, image_precision, image_recall])
            
        elif MODE == 'tracking':
            # TODO: Implement!
            pass
        
    # Write evaluation values into file.
    with open(evaluation_file, 'w') as output_file:
        for e in evaluation_values:
            if MODE == 'detection':
                print('%d,%.2f,%.2f' % (e[0], e[1], e[2]), file=output_file)
            elif MODE == 'tracking':
                # TODO: Implement!
                pass
            
            

print("Evaluation finished!")