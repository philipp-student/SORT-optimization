import socket, pickle
from PIL import Image
import numpy as np
from carla_camera_frame import CARLACameraFrame
import struct
import queue
from frame_source import ServerFrameSource, DirectoryFrameSource
from detector import YOLOv5
import cv2
import time
import os
import argparse
from sort import Sort

DETECTIONS_COLOR = (255, 0, 0)
TRACKERS_COLOR = (0, 255, 0)

FRAME_SOURCE = None

#OFFLINE_FRAME_DIRECTORY = r'D:\Philipp Student\HRW\Fahrassistenzsysteme 2\Seminararbeit\MOT15\test\AVG-TownCentre\img1'
#OFFLINE_FRAME_DIRECTORY = r'D:\Philipp Student\HRW\Fahrassistenzsysteme 2\Seminararbeit\MOT15\test\ETH-Jelmoli\img1'
#OFFLINE_FRAME_DIRECTORY = r'D:\Philipp Student\HRW\Fahrassistenzsysteme 2\Seminararbeit\MOT15\test\Venice-1\img1'
OFFLINE_FRAME_DIRECTORY = r'D:\Philipp Student\HRW\Fahrassistenzsysteme 2\Seminararbeit\MOT15\train\KITTI-17\img1'

HOST = 'localhost'
PORT = 50007

DETECTOR = None

OUTPUT_FILE_NAME = 'test_outputs'
OUTPUT_FILE_FOLDER = r'D:\Philipp Student\HRW\Repositories\fas-2-pietryga-student\output'
OUTPUT_FILE_EXTENSION = ".txt"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_FILE_FOLDER, "{0}{1}".format(OUTPUT_FILE_NAME, OUTPUT_FILE_EXTENSION))

# Converts normalized image coodinates to explicit image coordinates.
def norm2img(xyxy, width, height):
    return np.array([xyxy[0] * width, xyxy[1] * height, xyxy[2] * width, xyxy[3] * height])

# Plots a bounding box on the given image.
def plot_box(img, bounding_box_xyxy, color, thickness=2):        
    # Plot rectangle in image.
    return cv2.rectangle(img, (int(bounding_box_xyxy[0]), int(bounding_box_xyxy[1])), (int(bounding_box_xyxy[2]), int(bounding_box_xyxy[3])), color, thickness)

# Draws bounding boxes onto an image and displays it.
def display_bounding_boxes(img, bounding_boxes, color, normalized=True):
    if bounding_boxes is None: return
    
    # Draw each bounding box on image.
    num_boxes = bounding_boxes.shape[0]
    for i in range(num_boxes):        
        # Get coordinates of bounding box.
        bounding_box = bounding_boxes[i]
        
        # Compute explicit coordinates if they are normalized.
        if normalized:
            bounding_box = norm2img(bounding_box, img.shape[1], img.shape[0])
        
        # Plot bounding box.
        img = plot_box(img, bounding_box, color)

    # Display image.
    cv2.imshow('image', img)
    cv2.waitKey(1)

# Writes the given trackers into the given output file.
def write_trackers(output_file, trackers, frame_index):
    if trackers is None: return
    
    for t in trackers: 
        # Write tracker as xywh into file.                   
        print('%d,%d,%.2f,%.2f,%.2f,%.2f'%(frame_index,t[4],t[0],t[1],t[2]-t[0],t[3]-t[1]),file=output_file)

# Parses command line arguments.
def parse_args():
    """Parse input arguments."""
    # Instantiate command line argument parser.
    parser = argparse.ArgumentParser(description='Tracking')
    
    # Add arguments and descriptions.
    parser.add_argument('--mode', dest='mode', help='Operating mode. ONLINE if CARLA images should be used. OFFLINE if an offline dataset should be used.', 
                        type=str, default='OFFLINE')
    parser.add_argument('--display_tracks', dest='display_tracks', help='Whether the tracks should be displayed.',action='store_true')
    parser.add_argument('--display_detections', dest='display_detections', help='Whether the detections should be displayed.',action='store_true')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("--save_tracks", help="Whether the tracks should be saved.", action='store_true')
    parser.add_argument('--detector_type', dest='detector_type', help='Type of YOLO detector that should be used.', 
                        type=str, default='yolov5s')
    
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
    IOU_THRESHOLD = args.iou_threshold              # IOU threshold for association.
    SAVE_TRACKS = args.save_tracks                  # Whether the tracks should be saved in a file.
    DETECTOR_TYPE = args.detector_type              # Type of YOLO detector.
    
    print(">>> Initializing detector...")
    # Initialize detector.
    DETECTOR = YOLOv5(DETECTOR_TYPE)
    
    print(">>> Initializing frame source...")
    # Initialize frame source.
    if MODE == 'ONLINE':
        FRAME_SOURCE = ServerFrameSource(HOST, PORT)
    elif MODE == 'OFFLINE':
        FRAME_SOURCE = DirectoryFrameSource(OFFLINE_FRAME_DIRECTORY)
    else:
        print("Error: Unknown Mode")
        exit()
    
    print(">>> Initializing SORT tracker...")
    # Create instance of SORT tracker.
    mot_tracker = Sort(max_age=MAX_AGE, 
                       min_hits=MIN_HITS,
                       iou_threshold=IOU_THRESHOLD)
    
    print(">>> Initialization finished!")    
    print(">>> Tracking begins. Press Ctrl+C to stop the tracking.")
    while(True):
        
        # Open output file for writing.
        with open(OUTPUT_FILE_PATH,'w') as output_file:
        
            try:
                # Get frame from source.
                camera_frame = FRAME_SOURCE.get_frame()
                if camera_frame is None: continue                        
                
                # Detect pedestrians in frame.
                detections = DETECTOR.detect_pedestrians(camera_frame.frame)
                
                # Display frame and detected pedestrians within.
                if (DISPLAY_DETECTIONS):
                    display_bounding_boxes(camera_frame.frame, detections, DETECTIONS_COLOR)
                
                # TODO: Update trackers with detections and retrieve new trackers.
                # Format of returned tracker: [x1, y1, x2, y2, id].
                trackers = mot_tracker.update(detections)
                
                # Write updated tracker states into output file if desired.
                if (SAVE_TRACKS):
                    write_trackers(output_file, trackers, camera_frame.frame_index)
                
                # Display updated trackers if desired.
                if (DISPLAY_TRACKS):                
                    display_bounding_boxes(camera_frame.frame, trackers, TRACKERS_COLOR)
                
            except KeyboardInterrupt:
                # Clean up.
                print(">>> Cleaning up...")
                FRAME_SOURCE.cleanup()
                
                break

if __name__ == "__main__":    
    main()
    
    print(">>> Tracking stopped.")