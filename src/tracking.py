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

MODE = 'OFFLINE'
DISPLAY = True
SAVE_TRACKS = True

FRAME_SOURCE = None

#OFFLINE_FRAME_DIRECTORY = r'D:\Philipp Student\HRW\Fahrassistenzsysteme 2\Seminararbeit\MOT15\test\AVG-TownCentre\img1'
#OFFLINE_FRAME_DIRECTORY = r'D:\Philipp Student\HRW\Fahrassistenzsysteme 2\Seminararbeit\MOT15\test\ETH-Jelmoli\img1'
#OFFLINE_FRAME_DIRECTORY = r'D:\Philipp Student\HRW\Fahrassistenzsysteme 2\Seminararbeit\MOT15\test\Venice-1\img1'
OFFLINE_FRAME_DIRECTORY = r'D:\Philipp Student\HRW\Fahrassistenzsysteme 2\Seminararbeit\MOT15\train\KITTI-17\img1'

HOST = 'localhost'
PORT = 50007

DETECTOR_TYPE = 'yolov5s'
DETECTOR = None

OUTPUT_FILE_NAME = 'test_outputs'
OUTPUT_FILE_FOLDER = r'D:\Philipp Student\HRW\Repositories\fas-2-pietryga-student\output'
OUTPUT_FILE_EXTENSION = ".txt"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_FILE_FOLDER, "{0}{1}".format(OUTPUT_FILE_NAME, OUTPUT_FILE_EXTENSION))

# Converts normalized image coodinates to explicit image coordinates.
def norm2img(xyxy, width, height):
    return np.array([xyxy[0] * width, xyxy[1] * height, xyxy[2] * width, xyxy[3] * height])

# Plots a bounding box on the given image.
def plot_box(img, bounding_box_xyxy, color=(255, 0, 0), thickness=2):        
    # Plot rectangle in image.
    return cv2.rectangle(img, (int(bounding_box_xyxy[0]), int(bounding_box_xyxy[1])), (int(bounding_box_xyxy[2]), int(bounding_box_xyxy[3])), color, thickness)

# Draws bounding boxes onto an image and displays it.
def display_bounding_boxes(img, bounding_boxes, normalized=True):
    if bounding_boxes is None: return
    
    # Draw each bounding box on image.
    num_boxes = bounding_boxes.size(0)
    for i in range(num_boxes):        
        # Get coordinates of bounding box.
        bounding_box = bounding_boxes[i].numpy()
        
        # Compute explicit coordinates if they are normalized.
        if normalized:
            bounding_box = norm2img(bounding_box, img.shape[1], img.shape[0])
        
        # Plot bounding box.
        img = plot_box(img, bounding_box)

    # Display image.
    cv2.imshow('image', img)
    cv2.waitKey(1)

# Writes the given trackers into the given output file.
def write_trackers(output_file, trackers, frame_index):
    if trackers is None: return
    
    for t in trackers: 
        # Write tracker as xywh into file.                   
        print('%d,%d,%.2f,%.2f,%.2f,%.2f'%(frame_index,t[4],t[0],t[1],t[2]-t[0],t[3]-t[1]),file=output_file)

# Main function.
def main():
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
                if (DISPLAY):
                    display_bounding_boxes(camera_frame.frame, detections)
                
                # TODO: Update trackers with detections and retrieve new trackers.
                # Format of returned tracker: [x1, y1, x2, y2, id].
                trackers = None
                
                # Write updated tracker states into output file if desired.
                if (SAVE_TRACKS):
                    write_trackers(output_file, trackers, camera_frame.frame_index)
                
                # Display updated trackers if desired.
                if (DISPLAY):                
                    display_bounding_boxes(camera_frame.frame, trackers)
                
            except KeyboardInterrupt:
                # Clean up.
                print(">>> Cleaning up...")
                FRAME_SOURCE.cleanup()
                
                break

if __name__ == "__main__":    
    main()
    
    print(">>> Tracking stopped.")