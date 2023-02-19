"""
This file ... TODO
"""

import pandas as pd
import os.path as path

from loguru import logger
from sklearn.datasets import fetch_openml

from assignment_3.clustering.model import Dataset
from assignment_3.clustering.settings import TRAINING_DATA, TRAINING_LABELS, \
    TRAINING_LABELS_SMALL, TRAINING_DATA_SMALL, DATASET_DIR

data = path.join(DATASET_DIR, TRAINING_DATA)
labels = path.join(DATASET_DIR, TRAINING_LABELS)

data_s = path.join(DATASET_DIR, TRAINING_DATA_SMALL)
labels_s = path.join(DATASET_DIR, TRAINING_LABELS_SMALL)


def download_dataset():

    # Downloading data
    logger.info("Fetching data")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = y.astype(int)
    X = X / 255

    # Storing data
    d = Dataset(X, y)
    d.store(X_name=data, y_name=labels)


def read_dataset() -> Dataset:
    """
    Read datasets
    """
    logger.info("Reading datasets")
    return Dataset(
        x=pd.read_csv(data),
        y=pd.read_csv(labels).values.ravel()
    )


def read_small_dataset() -> Dataset:
    """
    Read small datasets
    """
    logger.info("Reading datasets")
    return Dataset(
        x=pd.read_csv(data_s),
        y=pd.read_csv(labels_s).values.ravel()
    )
