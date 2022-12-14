from typing import Tuple

import pandas as pd
import os.path as path

from loguru import logger
from sklearn.datasets import fetch_openml

from assignment_2.digits_classifiers.model import Dataset
from assignment_2.digits_classifiers.settings import DATASETS, TRAINING_DATA, TRAINING_LABELS, get_root_dir
from assignment_2.digits_classifiers.utils import create_dir

data = path.join(get_root_dir(), DATASETS, TRAINING_DATA)
labels = path.join(get_root_dir(), DATASETS, TRAINING_LABELS)

DATASET_DIR = path.join(get_root_dir(), DATASETS)


def download_dataset():

    # Creating Directory
    create_dir(DATASET_DIR)

    # Downloading data
    logger.info("Fetching data")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = y.astype(int)
    X = X / 255

    # Storing data
    logger.info("Saving data")
    X.to_csv(data, index=False)
    y.to_csv(labels, index=False)


def read_datasets() -> Dataset:
    """
    Read datasets
    """
    logger.info("Reading datasets")
    return Dataset(
        x=pd.read_csv(path.join(data)),
        y=pd.read_csv(labels).values.ravel()
    )
