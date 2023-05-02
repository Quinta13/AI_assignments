"""
This module provides a set of function for I/O operations
"""
from __future__ import annotations

import os.path as path
from os import path as path

import pandas as pd
from sklearn.datasets import fetch_openml

from clustering.globals import get_dataset_dir
from clustering.io_ import log
from clustering.model.model import Dataset
from clustering.settings import DATA, LABELS, \
    LABELS_SMALL, DATA_SMALL


""" DATASETS OPERATIONS """


def download_MNIST() -> Dataset:
    """
    Download MNIST dataset
    :return: MNIST dataset just downloaded
    """

    # Downloading data
    log("Fetching data")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = y.astype(int)
    X = X / 255

    # Storing data
    d = Dataset(x=X, y=y)
    return d


def _store_dataset(data: Dataset, x_name: str, y_name):
    """
    Store datasets using a STUB of Dataset method
    :param data: dataset to be stored
    :param x_name: name for data file
    :param y_name: name for labels file
    """
    data.store(x_name=x_name, y_name=y_name)


def store_MNIST(data: Dataset):
    """
    Store given dataset as MNIST one
    :param data: dataset to be stored
    """

    log("Storing MNIST")

    _store_dataset(data=data, x_name=DATA, y_name=LABELS)


def store_MNIST_SMALL(data: Dataset):
    """
    Store given dataset as small MNIST one
    :param data: dataset to be stored
    """

    log("Storing small MNIST")

    _store_dataset(data=data, x_name=DATA_SMALL, y_name=LABELS_SMALL)


def _read_dataset(x_name: str, y_name: str) -> Dataset:
    """
    Read dataset from proper directory
    :param x_name: name for data file
    :param y_name: name for labels file
    """

    x_file = path.join(get_dataset_dir(), f"{x_name}.csv")
    y_file = path.join(get_dataset_dir(), f"{y_name}.csv")

    log(f"Reading {x_file}")
    X = pd.read_csv(x_file)

    log(f"Reading {y_file}")
    y = pd.read_csv(y_file).values.ravel()

    return Dataset(
        x=X,
        y=y
    )


def read_MNIST() -> Dataset:
    """
    Read MNIST dataset in proper directory
    """
    log("Reading MNIST")

    return _read_dataset(DATA, LABELS)


def read_MNIST_small() -> Dataset:
    """
    Read small MNIST dataset in proper directory
    """

    log("Reading small MNIST")

    return _read_dataset(DATA_SMALL, LABELS_SMALL)
