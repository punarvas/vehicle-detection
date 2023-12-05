"""
CIS 579 Artificial Intelligence
Final Project

Vehicle detection using RCNN and Mask-RCNN
By: Akash Mahajan, Yash Dave

_________________________________________________________

# TODO: Add run instructions here

"""

# Import general packages
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from tqdm import tqdm

# Import local files
from modules.parser import *

# Root directory of the project
ROOT_DIR = os.getcwd()
DATASET_DIR = os.path.join(ROOT_DIR, "annotated_dataset")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# for cv selective search algorithm
cv2.setUseOptimized(True)
selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

############################################################
#  Util functions
############################################################
def SSProposedRegions(image): # get_rects(image)
    """
    This function take in an image and return selective Search algorithms
    proposed regions of interest
    """
    selective_search.setBaseImage(image)
    selective_search.switchToSelectiveSearchFast()
    rects = selective_search.process()
    return rects

def intersection_over_union(bBox1, bBox2):
    """
    This function takes in two bounding box regions defined by {x1, y1, width, heigth}
    and return overlap region (between 0 and 1).
    """
    assert bBox1['x1'] < bBox1['x2']
    assert bBox1['y1'] < bBox1['y2']
    assert bBox2['x1'] < bBox2['x2']
    assert bBox2['y1'] < bBox2['y2']

    x_left = max(bBox1['x1'], bBox2['x1'])
    y_top = max(bBox1['y1'], bBox2['y1'])
    x_right = min(bBox1['x2'], bBox2['x2'])
    y_bottom = min(bBox1['y2'], bBox2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bBox1['x2'] - bBox1['x1']) * (bBox1['y2'] - bBox1['y1'])
    bb2_area = (bBox2['x2'] - bBox2['x1']) * (bBox2['y2'] - bBox2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

############################################################
#  Dataset generation
############################################################
class DatasetGenerator:
    """
    This class helps generate the dataset to train RCNN model.
    It reads the images and annotations from the dataset path provided.
    """

    def __init__(self, datasetDir):
        self.datasetPath = datasetDir

    def loadDataset(self, subset):
        """
        loads the annotated training dataset from the given path.
        Subset: "ideal" or "non-ideal" dataset seperated by weather conditions
        """
        self.annotationsDir = os.path.join(self.datasetPath, subset, "annotations/instances_default.json")
        parser = Parser(self.annotationsDir)
        self.annotations = parser.group_annotations_by_image()

        self.imagesDir = os.path.join(self.datasetPath, subset, "images")
        self.images = parser.get_images()

        return True

    def saveRoIs(self, outputPath, IoUThreshold):
        """
        This function uses Selective Search Algorithm to genrate a bounding box 
        for true and false Regions of Interest. Both true and false RoI's 
        will be save in outputPath and will be used for training Resnet50 model
        outputPath: dir to save true and false RoI's
        IoUThreshold: Threshold for Intersection over Union for true and false regions
        """
        image_names = list(self.annotations.keys())
        true_target = os.path.join(outputPath, "true")
        false_target = os.path.join(outputPath, "false")
        
        if not os.path.exists(true_target):
            os.makedirs(true_target)
            print("Create true target output directory")
        if not os.path.exists(false_target):
            os.makedirs(false_target)
            print("Created false target output directory")

        target_shape = (224, 224) #resnet50 requires this dimensions
        bBox = 1
        for i in tqdm(range(len(image_names))):
            file_name = os.path.join(self.datasetPath, self.imagesDir, self.images[i]["file_name"])
            annotations = [annot["bbox"] for annot in self.annotations[image_names[i]]]
            
            image = cv2.imread(file_name)
            rects = SSProposedRegions(image)

            for i in range(len(annotations)):
                x1, y1, width, height = np.asarray(annotations[i], dtype=np.int32)  # format: (start_x, start_y, width, height)
                x2 = x1 + width
                y2 = y1 + height
                # true RoI
                roi = image[y1:y2, x1:x2]
                roi = cv2.resize(roi, target_shape)
                filename = str(bBox) + ".jpg"
                cv2.imwrite(os.path.join(outputPath, true_target, filename), roi)

                # false RoI
                SSbBox = 0
                false_images = []
                for each_rect in rects:
                    rx1, ry1, rx2, ry2 = each_rect
                    rx2 = rx1 + rx2
                    ry2 = ry1 + ry2
                    box1 = {"x1": x1, "y1": y1, "x2": x2, "y2": y2} # true annotation
                    box2 = {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2} # true annotation
                    iou = intersection_over_union(box1, box2)
                    # if IoU == 0.0, its a totally false image
                    if iou == 0.0:
                        if SSbBox != 5:
                            false_roi = image[ry1:ry2, rx1:rx2]
                            false_roi = cv2.resize(false_roi, target_shape)
                            false_images.append(false_roi)
                            # filename = str(bBox) + "_" + str(SSbBox) + ".jpg"
                            # cv2.imwrite(os.path.join(outputPath, false_target, filename), false_roi)
                            SSbBox += 1
                bBox += 1

        # Remove duplicate false images
        false_images_arr = np.asarray(false_images)
        flattened_arrays = false_images_arr.reshape(false_images_arr.shape[0], -1)
        # Find unique flattened arrays
        unique_indices = np.unique(flattened_arrays, axis=0, return_index=True)[1]
        # Keep only the unique arrays
        unique_false_images = false_images_arr[unique_indices]

        print("Found {} unique images out of {} false images.".format(len(unique_false_images), len(false_images)))
        for i in range(len(unique_false_images)):
            cv2.imwrite(os.path.join(outputPath, false_target, str(i + 1) + ".jpg"), unique_false_images[i])

        print("Exported {} RoI(s)".format(bBox))

        return True

if __name__ == '__main__':
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train R-CNN to detect vehicles.')
    parser.add_argument("command",
                        metavar="train or run",
                        help="'train' or 'run'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/vehicle/dataset/",
                        help='Directory of the Vehicle dataset')
    parser.add_argument('--subset', required=False,
                        metavar="'ideal' or 'non-ideal'",
                        help='wheter to train on ideal or non-ideal dataset')
    parser.add_argument('--logs', required=False,
                        default=LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path to image",
                        help='Image to apply the bounding box on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "run":
        assert args.image, "Provide --image to evaluate and apply bounding-box on"
    elif args.command == "train":
        assert args.subset, "Provide --subset to train on ideal or non-ideal dataset"

    # Load dataset from given dataset path
    if args.dataset is not None:
        mDataset = DatasetGenerator(args.dataset)
    else:
        mDataset = DatasetGenerator(DATASET_DIR)        
    
    print("Loading {} dataset from {}".format(args.subset, mDataset.datasetPath))
    mDataset.loadDataset(args.subset)

    print("Saving Selective Search recommended regions of interest to {}".format(OUTPUT_DIR))
    mDataset.saveRoIs(OUTPUT_DIR, 0.2)



