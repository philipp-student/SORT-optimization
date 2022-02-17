from frame_source import DirectoryFrameSource
from detector import YOLOv5
from tracking import norm2img, display_bounding_boxes
import numpy as np
import os

def convert_to_absolute(width, height, bounding_boxes):            
    for b_index, b in enumerate(bounding_boxes):
        bounding_boxes[b_index, :] = norm2img(b, width, height)
    return bounding_boxes

# Main folder with all datasets.
MAIN_FOLDER = r"D:\Philipp Student\HRW\Fahrassistenzsysteme 2\Seminararbeit\MOT15\train"

# Initialize YOLOv5 detector.
YOLO_DETECTOR = YOLOv5("yolov5s")

for dataset_folder in os.listdir(MAIN_FOLDER):
    
    # Dataset main folder.
    DATASET_MAIN_FOLDER = os.path.join(MAIN_FOLDER, dataset_folder)

    # Dataset images folder.
    OFFLINE_FRAME_DIRECTORY = os.path.join(DATASET_MAIN_FOLDER, "img1")

    # Name of dataset.
    dataset_name = dataset_folder

    # Detections file.
    detections_file = os.path.join(DATASET_MAIN_FOLDER, "det", "det.txt")

    # Custom detections file.
    custom_detections_file = os.path.join(DATASET_MAIN_FOLDER, "det", "det_custom.txt")

    # Groundtruth file.
    groundtruth_file = os.path.join(DATASET_MAIN_FOLDER, "gt", "gt.txt")

    # Custom groundtruth file.
    custom_groundtruth_file = os.path.join(DATASET_MAIN_FOLDER, "gt", "gt_custom.txt")

    # Output file for YOLOv5 detections.
    output_yolov5 = os.path.join(DATASET_MAIN_FOLDER, "det", "YOLOv5_" + dataset_name + ".txt")

    # Initialize frame source.
    FRAME_SOURCE = DirectoryFrameSource(OFFLINE_FRAME_DIRECTORY)

    ######### Detections with YOLOv5 #########
    
    # Inference on YOLOv5.
    num_frames = 0
    with open(output_yolov5,'w') as output_file:    
        while(num_frames < FRAME_SOURCE.num_frames):
            # Get image.
            img = FRAME_SOURCE.get_frame()        
            
            # Inference.
            detections = YOLO_DETECTOR.detect_pedestrians(img.frame)
            detections = detections.numpy()
            
            # Convert detection coordinates
            detections = convert_to_absolute(img.frame.shape[1], img.frame.shape[0], detections)
            
            # Write detections into file.
            for d in detections:
                print('%d,%.2f,%.2f,%.2f,%.2f' % (img.frame_index + 1, d[0], d[1], d[2], d[3]), file=output_file)
                
            num_frames += 1
            if num_frames % 10 == 0:
                print("%s: Frame %d" % (dataset_name, num_frames))

    # Clean frame source.
    FRAME_SOURCE.cleanup()

    ######### Create custom detection data #########
    
    # Load original detections from detections file.
    original_detections = np.loadtxt(detections_file, delimiter=',')
    
    # Convert from x1y1wh to xyxy by adding width to x1 and height to y1.
    original_detections[:, 4:6] += original_detections[:, 2:4]

    # Write custom detections file.
    with open(custom_detections_file, 'w') as output_file:
        for d in original_detections:            
            print('%d,%.2f,%.2f,%.2f,%.2f' % (d[0], d[2], d[3], d[4], d[5]), file=output_file)
     
    ######### Create custom groundtruth data #########
        
    # Load original groundtruths from groundtruth file.
    groundtruths = np.loadtxt(groundtruth_file, delimiter=',')

    # Convert from x1y1wh to xyxy by adding width to x1 and height to y1.
    groundtruths[:, 4:6] += groundtruths[:, 2:4]

    # Write custom detections file.
    with open(custom_groundtruth_file, 'w') as output_file:
        for gt in groundtruths:
            print('%d,%.2f,%.2f,%.2f,%.2f' % (gt[0], gt[2], gt[3], gt[4], gt[5]), file=output_file)

    print("Finished work for dataset %s!" % dataset_name)

print("Finished all the work!")