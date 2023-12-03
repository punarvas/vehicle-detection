"""
Mask-RCNN
Train a RCNN model for vehicle detection

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Yash Dave

______________________________________________________________

TODO: Provide usage info here
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from parser import *

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class VehicleConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Car"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + car

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class VehicleDataset(utils.Dataset):

    def load_vehicle(self, dataset_dir, subset):
        """Load a subset of the vehicle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("Car", 1, "Car")

        # Train or validation dataset?
        assert subset in ["ideal", "non_ideal"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations_path = os.path.join(dataset_dir, "annotations/instances_default.json")

        # Use parser to get images and bounding boxes
        mParser = Parser(annotations_path)

        # get annotations by image ID
        annotations = mParser.group_annotations_by_image()

        # Add images
        for a in range(len(annotations)):
            # Get the image id and bounding box coordinates

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, "images", mParser._images[a]["file_name"])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            bounding_boxes = []
            boxes = mParser.get_annotation_by_image_id(a)
            
            for box in boxes:
                bounding_boxes.append(box["bbox"])

            self.add_image(
                "Car",
                image_id=a,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                bounding_boxes=bounding_boxes)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, num of bbox] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a vehicle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Car":
            return super(self.__class__, self).load_mask(image_id)

        # Convert bounding box to a bitmap mask of shape
        # [height, width]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["bounding_boxes"])], dtype=np.uint8)

        # Set pixels inside bbox to 1
        i = 0
        for box in info["bounding_boxes"]:
            x = int(box[0])
            y = int(box[1])
            numRows = int(box[2])
            numCols = int(box[3])
            #rowIndex = np.arange(int(box[0]), int(box[2]), 1.0, dtype=int)
            rowIndex = [i for i in range(x, x + numRows)]
            # colIndex = np.arange(int(box[1]), int(box[3]), 1.0, dtype=int)
            colIndex = [i for i in range(y, y + numCols)]
            print(rowIndex)
            rr, cc = skimage.draw.polygon(colIndex, rowIndex)
            mask[rr, cc, i] = 1
            i += 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Car":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # ideal dataset.
    dataset_train = VehicleDataset()
    dataset_train.load_vehicle(args.dataset, "ideal")
    dataset_train.prepare()

    # non-ideal dataset
    dataset_val = VehicleDataset()
    dataset_val.load_vehicle(args.dataset, "non_ideal")
    dataset_val.prepare()

    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')
    

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect vehicles.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'bbox'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/vehicle/dataset/",
                        help='Directory of the Vehicle dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to initial weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the bounding box on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the bounding box on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "bbox":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = VehicleConfig()
    else:
        class InferenceConfig(VehicleConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "bbox":
        # detect_and_color_splash(model, image_path=args.image,
        #                         video_path=args.video)
        # TODO: Add function to apply bbox to image
        print("Use provided file to test model on image")
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'bbox'".format(args.command))