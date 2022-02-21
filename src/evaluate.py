import os
import numpy as np
from sort import associate_detections_to_trackers
from tracking import draw_bounding_boxes
from PIL import Image
import cv2
from helpers import xywh2xyxy

CORRESPONDENCE_IOU_THRESHOLD = 0.5
RECALL_INTERPOLATION_VALUES = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

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

def mean_average_precision(groundtruth_boxes, result_boxes):
    
    # Compute correspondences of groundtruth bounding boxes to result bounding boxes.
    matches, unmatched_groundtruths, unmatched_results = associate_detections_to_trackers(groundtruth_boxes, result_boxes[:, :4], 'iou', CORRESPONDENCE_IOU_THRESHOLD)
    
    # Create array to hold results. Format: [Is TP?, Confidence, Precision, Recall, Interpolated Precision]
    prec_rec = np.zeros((result_boxes.shape[0], 5))
    
    # Copy confidences.
    prec_rec[:, 1] = result_boxes[:, 4]
    
    # Check for each result if it's TP.
    for result_index in range(result_boxes.shape[0]):
        if result_index in unmatched_results:
            prec_rec[result_index, 0] = 0
        else:
            prec_rec[result_index, 0] = 1
    
    
    ### AT THE END ###
    # Rank values by confidence.
    prec_rec = prec_rec[prec_rec[:, 1].argsort()][::-1]
    
    # Compute precision and recall for each result.
    tp_counter = 0
    tp_overall = matches.shape[0]
    for i in range(prec_rec.shape[0]):
        if prec_rec[i, 0] == 1:
            tp_counter += 1
            
        # Compute precision.
        prec_rec[i, 2] = tp_counter / (i + 1)
        
        # Compute recall.
        try:
            prec_rec[i, 3] = tp_counter / tp_overall
        except ZeroDivisionError:
            prec_rec[i, 3] = 0    
    
    # Determine interpolated precision for each result.
    distinct_recall_values = np.unique(prec_rec[:, 3])    
    for recall_value in distinct_recall_values:
        
        # Get indices of columns that contain the current recall value.
        column_indices_with_recall = (prec_rec[:, 3] == recall_value).nonzero()
        
        # Select all columns that have the current recall value.
        columns_with_recall = prec_rec[column_indices_with_recall]
        
        # Select maximum precision of these columns.
        max_prec = columns_with_recall[:, 2].max()
        
        # Set the maximum precision as the interpolated precision.
        prec_rec[column_indices_with_recall, 4] = max_prec
    
    # Compute average precision.
    prec_sum = 0
    for recall_interpolation_value in RECALL_INTERPOLATION_VALUES:
        # Get columns that have a recall equal to or greater than the current interpolation value.
        relevant_columns = prec_rec[prec_rec[:, 3] >= recall_interpolation_value]
        
        # Check if there are any more relevant colums.
        if relevant_columns.shape[0] == 0:
            continue
        
        # Sort relevant columns by recall.
        relevant_columns = relevant_columns[relevant_columns[:, 3].argsort()]
        
        # Get reprsentative value for current interpolation value.
        representative = relevant_columns[0, 4]
        
        # Cumulate precision sum.
        prec_sum += representative
    
    return prec_sum / 11
        
def get_gt_id_index(correspondences, gt_id):
    for c_index, c in enumerate(correspondences):
        if c[0] == gt_id: return c_index
    
    return None

MODE = "tracking"
DETECTOR_TYPE = "yolov5"

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

    # Name of dataset.
    dataset_name = dataset_folder
    print("################# {0} #################".format(dataset_name))

    # Groundtruths file (custom format).
    groundtruths_file = os.path.join(GROUNDTRUTHS_FOLDER, "gt_custom.txt")

    # Orignal detections file.
    original_detections_file = os.path.join(DETECTIONS_FOLDER, "det_custom.txt")

    # YOLOv5 detections file.
    yolo_detections_file = os.path.join(DETECTIONS_FOLDER, "det_YOLOv5.txt")

    # Tracks file.
    tracks_file = os.path.join(TRACKS_FOLDER, "tracking_f1_yolov5s.txt")
    
    ############ Load data ############
    
    groundtruths = np.loadtxt(groundtruths_file, delimiter=',')
    original_detections = np.loadtxt(original_detections_file, delimiter=',')
    yolo_detections = np.loadtxt(yolo_detections_file, delimiter=',')
    tracks = np.loadtxt(tracks_file, delimiter=',')
    
    ############ Evaluation ############
    
    # Get number of frames in dataset.
    num_frames = int(np.max(groundtruths[:, 0]))
    
    # Helpers for detection evaluation.
    MAP_precision_recall_table = np.empty((0, 5), dtype=np.float64)
    MAP_tp_overall = 0
    
    # Helpers for tracking evaluation.
    MOTP_matches_sum = 0
    MOTP_error_sum = 0    
    MOTA_false_negatives_sum = 0
    MOTA_false_positives_sum = 0
    MOTA_id_switches_sum = 0
    MOTA_groundtruths_sum = groundtruths.shape[0]
    MOTA_correspondences = []
    
    for frame_index in range(1, num_frames + 1):
        
        # Get all groundtruth bounding boxes for current frame.
        groundtruth_boxes = groundtruths[groundtruths[:, 0] == frame_index, 1:]
        
        if MODE == 'detection':
            if DETECTOR_TYPE == 'original':
                # Get all original detection bounding boxes for current frame.
                detection_boxes = original_detections[original_detections[:, 0] == frame_index, 1:]
            elif DETECTOR_TYPE == 'yolov5':
                # Get all YOLOv5 detection bounding boxes for current frame.
                detection_boxes = yolo_detections[yolo_detections[:, 0] == frame_index, 1:]
            
            ############# Mean average precision #############            
            
            # Compute correspondences of groundtruth bounding boxes to result bounding boxes.
            matches, unmatched_groundtruths, unmatched_results = associate_detections_to_trackers(groundtruth_boxes[:, 1:], detection_boxes[:, :4], 'iou', CORRESPONDENCE_IOU_THRESHOLD)
            
            # Create temporary array to hold results. Format: [Is TP?, Confidence, Precision, Recall, Interpolated Precision]
            temp_precision_recall_table = np.zeros((detection_boxes.shape[0], 5))
            
            # Copy confidences.
            temp_precision_recall_table[:, 1] = detection_boxes[:, 4]
            
            # Check for each result if it's TP.
            for detection_index in range(detection_boxes.shape[0]):
                if detection_index in unmatched_results:
                    temp_precision_recall_table[detection_index, 0] = 0
                else:
                    temp_precision_recall_table[detection_index, 0] = 1
            
            # Update number of total matches.
            MAP_tp_overall += matches.shape[0]
            
            # Append evaluation result to table.
            MAP_precision_recall_table = np.vstack([MAP_precision_recall_table, temp_precision_recall_table])                        
            
        elif MODE == 'tracking':    
            # Get all tracks of current frame.            
            track_boxes = tracks[tracks[:, 0] == frame_index, 1:]
  
            # Compute correspondences of groundtruth bounding boxes to track bounding boxes. Include iou of each match for MOTP computation.
            matches, unmatched_groundtruths, unmatched_tracks = associate_detections_to_trackers(groundtruth_boxes[:, 1:], track_boxes[:, 1:], 'iou', CORRESPONDENCE_IOU_THRESHOLD, True)
    
            ############# Multi Object Tracking Precision (MOTP) #############
                
            # Determine number of matches and add it to the sum.
            MOTP_matches_sum += matches.shape[0]
        
            # Compute error for each match and add it to the sum.
            for match in matches:
                MOTP_error_sum += 1 - match[2]
                
            ############# Multi Object Tracking Accuracy (MOTA) #############
            
            # Determine number of false negatives and add it to the sum.
            MOTA_false_negatives_sum += unmatched_groundtruths.shape[0]
            
            # Determine number of false positives and add it to the sum.
            MOTA_false_positives_sum += unmatched_tracks.shape[0]
                        
            # Determine number of id switches.
            num_id_switches = 0
            
            for match in matches:
                gt_id = groundtruth_boxes[int(match[0]), 0]
                track_id = track_boxes[int(match[1]), 0]
                
                gt_id_index = get_gt_id_index(MOTA_correspondences, gt_id)
                if gt_id_index == None:
                    # Add match to correspondences.
                    MOTA_correspondences.append(np.array([gt_id, track_id]))
                else:
                    # Get correspondence with current groundtruth id.
                    existing_correspondence = MOTA_correspondences[gt_id_index]
                    
                    # Check whether the groundtruth object is now tracked by a new tracker.
                    if existing_correspondence[1] != track_id:
                        # Increment id switch sum.
                        num_id_switches += 1
                        
                        # Remove old correspondence.
                        MOTA_correspondences.pop(gt_id_index)
                        
                        # Add match as new correspondence.
                        MOTA_correspondences.append(np.array([gt_id, track_id]))

            # Add number of id switches to sum.
            MOTA_id_switches_sum += num_id_switches

    # Postprocessing of collected evaluation values.
    if MODE == 'detection':        
        ############# Mean average precision #############            
        
        # Rank values by confidence.
        MAP_precision_recall_table = MAP_precision_recall_table[MAP_precision_recall_table[:, 1].argsort()][::-1]

        # Compute precision and recall for each result.
        tp_counter = 0
        for i in range(MAP_precision_recall_table.shape[0]):
            if MAP_precision_recall_table[i, 0] == 1:
                tp_counter += 1
        
            # Compute precision.
            MAP_precision_recall_table[i, 2] = tp_counter / (i + 1)
            
            # Compute recall.
            try:
                MAP_precision_recall_table[i, 3] = tp_counter / MAP_tp_overall
            except ZeroDivisionError:
                MAP_precision_recall_table[i, 3] = 0    
        
        # Determine interpolated precision for each result.
        distinct_recall_values = np.unique(MAP_precision_recall_table[:, 3])    
        for recall_value in distinct_recall_values:        
            # Get indices of columns that contain the current recall value.
            column_indices_with_recall = (MAP_precision_recall_table[:, 3] == recall_value).nonzero()
            
            # Select all columns that have the current recall value.
            columns_with_recall = MAP_precision_recall_table[column_indices_with_recall]
            
            # Select maximum precision of these columns.
            max_prec = columns_with_recall[:, 2].max()
            
            # Set the maximum precision as the interpolated precision.
            MAP_precision_recall_table[column_indices_with_recall, 4] = max_prec
        
        # Compute average precision.
        average_precision_sum = 0
        for recall_interpolation_value in RECALL_INTERPOLATION_VALUES:
            # Get columns that have a recall equal to or greater than the current interpolation value.
            relevant_columns = MAP_precision_recall_table[MAP_precision_recall_table[:, 3] >= recall_interpolation_value]
            
            # Check if there are any more relevant colums.
            if relevant_columns.shape[0] == 0:
                continue
            
            # Sort relevant columns by recall.
            relevant_columns = relevant_columns[relevant_columns[:, 3].argsort()]
            
            # Get representative value for current interpolation value.
            representative = relevant_columns[0, 4]
            
            # Cumulate precision sum.
            average_precision_sum += representative
        
        average_precision = average_precision_sum / 11

    # Print evaluation results.
    if MODE == 'detection':        
        print("Mean average precision: {0}".format(average_precision))
    elif MODE == 'tracking':
        print("Multi Object Tracking Accuracy (MOTA): {0} (m: {1}, fp: {2}, mme: {3}, gt: {4})".format(1 - ((MOTA_false_negatives_sum + MOTA_false_positives_sum + MOTA_id_switches_sum) / MOTA_groundtruths_sum), 
              MOTA_false_negatives_sum, MOTA_false_positives_sum, MOTA_id_switches_sum, MOTA_groundtruths_sum))
        print("Multi Object Tracking Precision (MOTP): {0} (d: {1}, c: {2})".format(MOTP_error_sum / MOTP_matches_sum, MOTP_error_sum, MOTP_matches_sum))
    
print("Evaluation finished!")