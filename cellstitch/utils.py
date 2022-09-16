import numpy as np


def get_lbls(mask):
    """
    returns the number of labels in the mask, including the background label.
    """
    return np.unique(mask)


def is_empty(mask):
    """
    return if the frame is empty.
    """
    return len(mask.get_lbls()) == 1
