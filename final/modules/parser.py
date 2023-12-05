import json
import random
import pandas as pd


class Parser:
    def __init__(self, file_path: str):
        self._file_path = file_path
        try:
            file = open(file_path, "r", encoding="utf8")
            self._json = json.load(file)
            file.close()
        except FileNotFoundError:
            raise Exception(f"Annotations at {file_path} does not exists")
        except OSError:
            raise Exception("Fatal OS Error. Cannot read file")
        
        self._images = self._json["images"]
        self._annotations = self._json["annotations"]
        self._nimages = len(self._images)
        self._nannot = len(self._annotations)
    
    def group_annotations_by_image(self):
        """
        Picks up each image from `images` list and finds all associated annotations in `annotations` list
        images: a list of image metadata in CVAT annotation file.
        annotations: a list of annotations in CVAT annotation file.
        returns: a dictionary of annotations grouped by image IDs
        """
        self._group = {}
        if (self._group != {}):
            return self._group
        for i in range(self._nimages):
            image_id = self._images[i]["id"]
            self._group[image_id] = []
            for j in range(self._nannot):
                if self._annotations[j]["image_id"] == image_id:
                    self._group[image_id].append(self._annotations[j])
        
        return self._group
    
    def get_annotation_by_image_id(self, image_id:int):
        """
        Returns annotations for a given image
        image_id: a unique ID for the image
        annotations: a set of annotations which are usually accessed by key "annotations" in CVAT annotation file
        returns: A list of annotations for a single image file
        """
        annotation_list = []
        for i in range(self._nannot):
            if self._annotations[i]["image_id"] == image_id:
                annotation_list.append(self._annotations[i])
        
        return annotation_list
    
    def get_annotation_by_image_name(self, image_name: str):
        image_id = None
        for i in range(self._nimages):
            if self._images[i]["file_name"] == image_name:
                image_id = self._images[i]["id"]
                break

        if image_id != None:
            return self.get_annotation_by_image_id(image_id)
        else:
            return None

    def get_json(self):
        return self._json

    def get_images(self):
        return self._images

    def get_annotations(self):
        return self._annotations

    def to_pandas(self, save_to_file: bool = False):
        """
        TODO: Write code to save the grouped annotations to CSV file
        """