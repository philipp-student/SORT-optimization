import socket, pickle
from PIL import Image
import numpy as np
from camera_frame import CameraFrame
import struct
import queue
from frame_source import ServerFrameSource, DirectoryFrameSource
from detector import YOLOv5
import cv2
import time
import os
import argparse
from sort import Sort
import time
import helpers

# Colors of bounding boxes.
DETECTIONS_COLOR = (255, 0, 0)
TRACKERS_COLOR = (0, 255, 0)

FRAME_SOURCE = None
DETECTOR = None

# Converts normalized image coodinates to explicit image coordinates.
def norm2img(xyxy, width, height):    
    new_x1 = xyxy[:, 0] * width
    new_y1 = xyxy[:, 1] * height
    new_x2 = xyxy[:, 2] * width
    new_y2 = xyxy[:, 3] * height
    
    return np.concatenate([new_x1.reshape(-1, 1), new_y1.reshape(-1, 1), new_x2.reshape(-1, 1), new_y2.reshape(-1, 1)], axis=1)

# Draws a bounding box on the given image.
def draw_boxes(img, bounding_boxes_xyxy, color, thickness=2):   
    for b in bounding_boxes_xyxy:             
        # Plot bounding box as rectangle in image.
        img = cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, thickness)
    
    return img

# Draws bounding boxes onto an image and displays it.
def display_bounding_boxes(img, bounding_boxes, color, normalized=True):
    if bounding_boxes is None: return
    
    # Compute absolute coordinates if they are normalized.
    if normalized:
        bounding_boxes = norm2img(bounding_boxes, img.shape[1], img.shape[0])
    
    # Draw bounding boxes onto image.
    img = draw_boxes(img, bounding_boxes, color)
    
    # Convert image to RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display image.    
    cv2.imshow('image', img)
    cv2.waitKey(1)

# Draws bounding boxes onto an image withou displaying it.
def draw_bounding_boxes(img, bounding_boxes, color, normalized=True):
    if bounding_boxes is None: return
    
    # Draw each bounding box on image.
    num_boxes = bounding_boxes.shape[0]
    for i in range(num_boxes):        
        # Get coordinates of bounding box.
        bounding_box = bounding_boxes[i]
        
        # Compute explicit coordinates if they are normalized.
        if normalized:
            bounding_box = norm2img(bounding_box.reshape(1, -1), img.shape[1], img.shape[0])
        
        # Draw bounding box onto image.
        img = draw_box(img, bounding_box, color)

    # Convert image to RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

# Parses command line arguments.
def parse_args():
    """Parse input arguments."""
    # Instantiate command line argument parser.
    parser = argparse.ArgumentParser(description='Tracking')
    
    # Add arguments and descriptions.
    parser.add_argument('--mode', dest='mode', help='Operating mode. "online" if CARLA images should be used. "offline" if an offline dataset should be used.', 
                        type=str, default='offline')
    parser.add_argument('--display_tracks', dest='display_tracks', help='Whether the tracks should be displayed.',action='store_true')
    parser.add_argument('--display_detections', dest='display_detections', help='Whether the detections should be displayed.',action='store_true')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--cost_type", help="Type of cost for match.", type=str, default='iou')
    parser.add_argument("--cost_threshold", help="Minimum cost for match.", type=float, default=0.3)
    parser.add_argument('--detector_type', dest='detector_type', help='Type of YOLO detector that should be used.', 
                        type=str, default='yolov5s')
    parser.add_argument('--frame_source_host', dest='frame_source_host', help='IP Address of host that provides the CARLA frames.', 
                        type=str, default='localhost')
    parser.add_argument("--frame_source_port", 
                        help="Port of host that provides the CARLA frames.", 
                        type=int, default=50007)
    parser.add_argument('--dataset', dest='dataset', help='Dataset on which the tracking shoould be performed.', 
                        type=str, default='ADL-Rundle-6')
    
    # Parse given arguments and return results.
    return parser.parse_args()

# Main function.
def main():
    # Parse command line arguments.
    args = parse_args()
    
    # Set properties of tracking process.    
    MODE = args.mode                                # Tracking mode
    DISPLAY_DETECTIONS = args.display_detections    # Whether detections should be displayed.
    DISPLAY_TRACKS = args.display_tracks            # Whether tracks should be displayed.
    MAX_AGE = args.max_age                          # Maximum age of a track without getting detections.
    MIN_HITS = args.min_hits                        # Minimum detections for a track to be created.
    COST_TYPE = args.cost_type                      # Cost type for association.
    COST_THRESHOLD = args.cost_threshold            # Cost threshold for association.
    DETECTOR_TYPE = args.detector_type              # Type of YOLO detector.
    HOST = args.frame_source_host                   # IP Address of host that provides CARLA frames.
    PORT = args.frame_source_port                   # Port of host that provides CARLA frames.
    DATASET_NAME = args.dataset                     # Dataset on which tracking should be performed.
    
    # Initialize time recording.
    total_time_detection = 0.0
    total_time_tracking = 0.0
    total_frames = 0
    
    print(">>> Initializing detector...")
    # Initialize detector.
    DETECTOR = YOLOv5(DETECTOR_TYPE)
    
    print(">>> Initializing frame source...")
    # Initialize frame source.
    if MODE == 'online':
        FRAME_SOURCE = ServerFrameSource(HOST, PORT)
    elif MODE == 'offline':
        offline_directory = os.path.join(helpers.DATA_DIRECTORY, DATASET_NAME, "img1")
        
        if not os.path.exists(offline_directory):
            print("Unknown dataset {0}".format(DATASET_NAME))
            exit()
        
        FRAME_SOURCE = DirectoryFrameSource(offline_directory)
    else:
        print("Error: Unknown mode {0}".format(MODE))
        exit()
    
    print(">>> Initializing SORT tracker...")
    # Create instance of SORT tracker.
    mot_tracker = Sort(max_age=MAX_AGE, 
                       min_hits=MIN_HITS,
                       cost_type=COST_TYPE,
                       cost_threshold=COST_THRESHOLD)
    
    print(">>> Initialization finished!")    
    print(">>> Realtime tracking begins. Press Ctrl+C to stop.")        
        
    while total_frames < FRAME_SOURCE.num_frames:            
        try:
            # Get frame from source.
            camera_frame = FRAME_SOURCE.get_frame()
            if camera_frame is None: continue                 
            
            ######## DETECTION ########   
            start_time_detection = time.time()

            # Detect pedestrians in frame.
            detections = DETECTOR.detect_pedestrians(camera_frame.frame)
            
            cycle_time_detection = time.time() - start_time_detection
            total_time_detection += cycle_time_detection
                        
            # Display frame and detected pedestrians within if desired.
            if DISPLAY_DETECTIONS:
                display_bounding_boxes(camera_frame.frame, detections, DETECTIONS_COLOR)
            
            ######## TRACKING ########                
            start_time_tracking = time.time()
            
            # Update trackers with detections and retrieve new trackers.
            # Format of returned tracker: [x1, y1, x2, y2, id].
            trackers = mot_tracker.update(detections)
                            
            cycle_time_tracking = time.time() - start_time_tracking
            total_time_tracking += cycle_time_tracking                
            ##########################
            
            # Increment frame count.
            total_frames += 1
            
            # Display updated trackers if desired.
            if DISPLAY_TRACKS:
                display_bounding_boxes(camera_frame.frame, trackers, TRACKERS_COLOR)
            
        except KeyboardInterrupt:                
            break
            
    # Clean up.
    print(">>> Cleaning up...")
    FRAME_SOURCE.cleanup()
    
    # Runtime analysis results for detection.
    detection_fps = total_frames / total_time_detection
    detection_mspf = (total_time_detection / total_frames) * 1000.0
    print("Detection took %.3f seconds for %d frames (%.1f FPS | %.3f ms/frame)" % 
          (total_time_detection, total_frames, detection_fps, detection_mspf))
    
    # Runtime analysis results for tracking.
    tracking_fps = total_frames / total_time_tracking
    tracking_mspf = (total_time_tracking / total_frames) * 1000.0
    print("Tracking took %.3f seconds for %d frames (%.1f FPS | %.3f ms/frame)" % 
          (total_time_tracking, total_frames, tracking_fps, tracking_mspf))
    
if __name__ == "__main__":    
    main()    
    print(">>> Tracking stopped.")