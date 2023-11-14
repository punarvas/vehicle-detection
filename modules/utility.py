import numpy as np
from cv2.ximgproc.segmentation import createSelectiveSearchSegmentation as SelectiveSearch


class Activation:
    """
    This class is intended to filter out those region proposals whose IoU with original bounding box
    does not satisfy the given threshold value.
    """
    def __init__(self, threshold: np.float32, labels: list = [1, 0]):
        self._threshold = threshold
        self._labels = labels  # in the format [true, false]
        self.label = lambda iou: self._labels[0] if iou > self._threshold else self._labels[0]

    def get_threshold(self):
        return self._threshold

    def set_labels(self, labels: list):
        assert len(labels) == 2
        self._labels = labels

    def get_labels(self):
        return {"true": self._labels[0], "false": self._labels[1]}

    def revert_labels(self):
        self._labels = [1, 0] 


class RegionProposal:
    """
    Generate region proposals for the image.
    Define IoU threshold to select best region proposals and drop those does not satisfy the threshold
    Call top() function to select top proposals that satisfy the IoU threshold
    """
    def __init__(self, image: np.ndarray, annotations: np.ndarray, iou_threshold: np.float32 = 0.5):
        self._image = image
        self._annotations = annotations
        self._activation = Activation(threshold = iou_threshold)

    def iou(self, t: np.ndarray, p: np.ndarray):
        """
        Calculate the IoU of t and p. `t` is ground truth and `p` is proposal.
        """
        assert t.shape == p.shape
        # Find bounding box coordinate of intesection (i)
        i_x1 = max(t[0], p[0])
        i_y1 = max(t[1], p[1])
        i_x2 = min(t[2], p[2])
        i_y2 = min(t[3], p[3])
        # Area of intersection bounding box
        i_area = (i_x2 - i_x1) * (i_y2 - i_y1)
        # truth and prediction areas
        t_area = (t[2] - t[0]) * (t[3] - t[1])
        p_area = (p[2] - p[0]) * (p[3] - p[1])
        # Measure
        u_area = t_area + p_area - i_area
        iou = round(i_area / u_area, 2)
        return iou

    def propose(self):
        """
        Generate region proposals for each image in the `image` array.
        TODO: Highly memory consuming task. Can be improved?
        """
        selective_search = SelectiveSearch()
        selective_search.setBaseImage(self._image)
        selective_search.switchToSelectiveSearchQuality()
        self._bounding_proposals = selective_search.process()

        self._region_proposals = []
        for region in self._bounding_proposals:
            x, y, width, height = region
            self._region_proposals.append(self._image[y:y+height, x:x+width])

    def top(self, n:int = 0):
        """
        Returns the to `n` region proposals that satisfy the IoU threshold.
        It is not gauranteed that exactly `n` proposals will be returned because there is a chance
        that number of region proposals that satisfy the IoU are fewer than `n`.
        """
        best_proposals = []
        k = 0
        for t in self._annotations:
            for p in self._bounding_proposals:
                if k == n and n != 0:
                    break
                iou = self.iou(t, p)
                label = self._activation.label(iou)
                if label == 1:
                    best_proposals.append((p, iou))
                    k += 1
        return best_proposals

    def get_proposals(self):
        """
        Return the region proposals for the image
        """
        try:
            return self._region_proposals
        except:
            self.propose()
            return self._region_proposals
        
    def get_warped_proposals(self, target_shape: set = (227, 227), p: int = 8):
        """
        Warp the image into a target shape.
        This is useful for training purpose when we train the R-CNN on a backbone like VGG16
        and ResNet50.
        """
        warped_proposals = []
        for region in self._region_proposals:
            original_h, original_w = region.shape[:2]
            new_h = original_h + (p * 2)
            new_w = original_w + (p * 2)
            padded_region = np.full((new_h, new_w, 3), (0, 0, 0))   # (0, 0, 0) padding with black color
            padded_region[p:p+original_h, p:p+original_w] = region
            warped_proposals.append(padded_region)
        return np.asarray(warped_proposals)
