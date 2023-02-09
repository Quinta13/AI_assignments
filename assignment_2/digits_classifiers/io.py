import os

import numpy as np
import pandas as pd
import os.path as path

from PIL import ImageOps, Image
from loguru import logger
from sklearn.datasets import fetch_openml

from assignment_2 import Dataset
from assignment_2 import DATASETS, TRAINING_DATA, TRAINING_LABELS, get_root_dir, IN_IMAGES
from assignment_2 import create_dir

DATASET_DIR = path.join(get_root_dir(), DATASETS)

data = path.join(DATASET_DIR, TRAINING_DATA)
labels = path.join(DATASET_DIR, TRAINING_LABELS)

fool = path.join(DATASET_DIR, "fool.csv")




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


def save_fool_dataset():
    """
    Saves fool images as a dataframe
    """

    logger.info(f"Saving fool dataset {fool}")

    img_dir = os.path.join(get_root_dir(), IN_IMAGES)

    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(img_dir)
              for f in filenames]

    arr_images = [
        (1 - (np.array(ImageOps.grayscale(Image.open(image))) / 255)).flatten()
        for image in images
    ]

    pixels = {
        f"pixel{i+1}": [arr[i] for arr in arr_images]
        for i in range(len(arr_images[0]))
    }

    df = pd.DataFrame(pixels)

    df.to_csv(fool, index=False)


def read_fool_dataset() -> pd.DataFrame:
    """
    :return: fool dataset
    """
    return pd.read_csv(fool)