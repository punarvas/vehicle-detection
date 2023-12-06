import numpy as np


class Activation:
    def __init__(self, threshold):
        self._threshold = threshold
        self.label = lambda iou: 1 if iou > self._threshold else 0


def iou(t: np.ndarray, p: np.ndarray):  # t = truth, p = prediction / region proposal
    # Find bounding box coordinate of intesection (i)
    i_x1 = np.max([t[0], p[0]])
    i_y1 = np.max([t[1], p[1]])
    i_x2 = np.min([t[2], p[2]])
    i_y2 = np.min([t[3], p[3]])
    # Area of intersection bounding box
    i_area = (i_x2 - i_x1) * (i_y2 - i_y1)
    # truth and prediction areas
    t_area = (t[2] - t[0]) * (t[3] - t[1])
    p_area = (p[2] - p[0]) * (p[3] - p[1])
    # Measure
    u_area = t_area + p_area - i_area
    iou = round(i_area / u_area, 2)
    return iou 