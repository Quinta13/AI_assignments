"""
This file ... TODO
"""

import os.path as path


def get_root_dir() -> str:
    """
    Returns the path to the root of directory project.
    :return: string representing the dir path
    """

    # Remember that the relative path here is relative to __file__,
    # so an additional ".." is needed
    return str(path.abspath(path.join(__file__, "../")))


DATASETS = "datasets"
IMAGES = "images"

DATASET_DIR = path.join(get_root_dir(), DATASETS)
IMAGES_DIR = path.join(get_root_dir(), IMAGES)

TRAINING_DATA = "pixels.csv"
TRAINING_LABELS = "labels.csv"

TRAINING_DATA_SMALL = "pixels_s.csv"
TRAINING_LABELS_SMALL = "labels_s.csv"
