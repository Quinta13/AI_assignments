import pandas as pd
import os.path as path

from loguru import logger
from sklearn.datasets import fetch_openml

from clustering.model import Dataset
from clustering.settings import get_root_dir, DATASETS, TRAINING_DATA, TRAINING_LABELS
from clustering.utils import create_dir

DATASET_DIR = path.join(get_root_dir(), DATASETS)

data = path.join(DATASET_DIR, TRAINING_DATA)
labels = path.join(DATASET_DIR, TRAINING_LABELS)


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
        x=pd.read_csv(data),
        y=pd.read_csv(labels).values.ravel()
    )
