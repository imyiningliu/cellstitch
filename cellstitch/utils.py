import os
import numpy as np


def get_lbls(mask):
    """
    Returns the number of labels in the mask, including the background label.
    """
    return np.unique(mask)


def is_empty(mask):
    """
    Return if the frame is empty.
    """
    return len(get_lbls(mask)) == 1


def get_filenames(data_path):
    """
    Returns the .npy filenames from data_path.
    """

    filenames = []

    for file in os.listdir(data_path):
        if file.endswith(".npy"):
            filenames.append(file)
    return filenames
