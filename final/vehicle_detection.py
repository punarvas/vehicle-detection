"""
CIS 579 Artificial Intelligence
Final Project

Vehicle detection using RCNN and Mask-RCNN
By: Akash Mahajan, Yash Dave

_________________________________________________________

# Commands to train and run

1. Training the model
    a. Ideal dataset: 'python3 vehicle_detection.py train --subset=ideal' 
    b. Non-ideal dataset: 'python3 vehicle_detection.py train --subset=non_ideal'

2. Running the model to predict bounding boxes on given image
"python3 vehicle_detection.py run --image=/path/to/image.jpg"

"""

# Import general packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from tqdm import tqdm
import keras as KE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

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

def encode(label: int):
    """
    Utility function to encode labels
    """
    code = [0, 0]
    index = 0 if label == 0 else 1
    code[index] = 1.0
    return np.array(code)

def get_region(image, rect: [], target_shape: set = (224, 224)):
    """
    Helper function to regions list of traget_shape
    """
    x1, y1, x2, y2 = rect
    region = image[y1:y2, x1:x2]
    region = cv2.resize(region, target_shape)
    return region

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
            train_images = []
            train_labels = []

            for i in range(len(annotations)):
                x1, y1, width, height = np.asarray(annotations[i], dtype=np.int32)  # format: (start_x, start_y, width, height)
                x2 = x1 + width
                y2 = y1 + height
                # true RoI
                roi = image[y1:y2, x1:x2]
                roi = cv2.resize(roi, target_shape)
                filename = str(bBox) + ".jpg"
                cv2.imwrite(os.path.join(outputPath, true_target, filename), roi)
                train_images.append(roi)
                train_labels.append(1)

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
                            train_images.append(false_roi)
                            train_labels.append(0)
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
            
        print(false_images_arr.shape)
        print(unique_false_images.shape)
        print("Exported {} RoI(s)".format(bBox))

        return train_images, train_labels

############################################################
#  Building and training model
############################################################   
class TrainModel:
    """
    Class to build and train RCNN model
    """ 

    def splitTrainAndTestData(self, iamges, labels, testSize):
        """
        Function to split the data into train and test dataset for model
        """
        self.images = images
        self.labels = labels
        self.encoded_labels = np.array(list(map(encode, self.labels)))

        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(self.images, self.encoded_labels, test_size = testSize)
        self.test_images, self.val_images, self.test_labels, self.val_labels = train_test_split(self.test_images, self.test_labels, test_size = testSize)

    
    def initResNet50Model(self, input_shape):
        """
        Function to initialize a ResNet50 model with given input_shape
        """
        self.input_shape = input_shape
        self.model = ResNet50(weights = "imagenet", include_top = False, input_shape = self.input_shape)
        for layer in self.model.layers:
            layer.trainable = False

    def groupLayers(self):
        """
        Function to group model layers
        """
        self.rcnn = Sequential([self.model, Flatten(), Dense(256, activation = "relu"), Dropout(0.5), Dense(2, activation = "softmax")], name="RCNN")
        self.rcnn.compile(optimizer = Adam(learning_rate = 0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
        self.rcnn.summary()

    def fitModel(self):
        """
        Function to fit and build rcnn model
        """
        self.history = self.rcnn.fit(self.train_images, self.train_labels, epochs = 10, batch_size = 8, validation_data = (self.val_images, self.val_labels))
        self.rcnn.save("vehicles_rcnn.h5")
        self.y_hat = self.rcnn.predict(self.test_images)
        self.rcnn.save("vehicles_rcnn.keras")

    def predictBoundingBox(self, imagePath, modelPath="vehicles_rcnn.keras"):
        # Predict bounding boxes on given image
        rcnn = tf.keras.models.load_model("vehicles_rcnn.keras")
        sample_image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        rects = SSProposedRegions(sample_image)

        for rect in rects:
            x1, y1, width, height = rect
            x2 = x1 + width
            y2 = y1 + height
            # cv2.rectangle(copy, (x1, y1), (x2, y2), (255, 255, 0))

        indices = np.random.randint(0, len(rects), 100)
        input_rects = []

        for i in indices:
            x1, y1, width, height = rects[i]
            x2 = x1 + width
            y2 = y1 + height
            input_rects.append([x1, y1, x2, y2])

        regions = []
        for rect in input_rects:
            regions.append(get_region(image, rect))

        regions = np.array(regions).reshape((-1, 224, 224, 3))
        print(regions.shape)

        predict = rcnn.predict(regions)
        predict = np.array(list(map(np.argmax, predict)))

        # Find rects that contains rectangles
        true_indices = np.where(predict == 1)[0]

        for i in true_indices:
            x1, y1, x2, y2 = input_rects[i]
            cv2.rectangle(sample_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

        plt.imshow(sample_image)
        plt.axis("off")
        plt.show()

############################################################
#  Main
############################################################
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

    # if run command passed output predicted bounding boxes and return
    if args.command == "run":
        mModel = TrainModel()
        mModel.predictBoundingBox(args.image)
        sys.exit()

    # Load dataset from given dataset path
    if args.dataset is not None:
        mDataset = DatasetGenerator(args.dataset)
    else:
        mDataset = DatasetGenerator(DATASET_DIR)        
    
    print("Loading {} dataset from {}".format(args.subset, mDataset.datasetPath))
    mDataset.loadDataset(args.subset)

    print("Saving Selective Search recommended regions of interest to {}".format(OUTPUT_DIR))
    images, labels = mDataset.saveRoIs(OUTPUT_DIR, 0.2)
    
    images = np.array(images)
    labels = np.array(labels)

    # Create model instance
    mModel = TrainModel()

    # split training and testing image set
    mModel.splitTrainAndTestData(images, labels, 0.2)

    print(mModel.train_images.shape)
    print(mModel.train_labels.shape)

    # Initialize model
    print("Building ResNet50 model")
    input_shape = (224, 224, 3) # required by ResNet50
    mModel.initResNet50Model(input_shape)

    # Group layers and fit model
    mModel.groupLayers()

    # Build and fit model
    mModel.fitModel()