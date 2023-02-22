"""
This file ... TODO
"""

from __future__ import annotations

import numpy as np
import os
import pandas as pd
from typing import List, Dict

from loguru import logger
from matplotlib import pyplot as plt

from assignment_3.clustering.settings import IMAGES_DIR

SIZE = 28


def create_dir(path: str, log: bool = True):
    """
    Create directory if doesn't exists
    :param path: directory path
    :param log: activate logging
    """
    try:
        if log:
            logger.info(f"Creating {path}")
        os.makedirs(path)
    except FileExistsError:
        if log:
            logger.info(f"{path} already exists")


def chunks(lst: List, n: int) -> np.array:
    """
    Split given list in a matrix with fixed-length rows length
    :param lst: list to split
    :param n: length of sublist
    :return: matrix with n rows
    """

    def chunks_():
        """
        Auxiliary function to exploit yield operator property
        """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    list_len = len(lst)

    # list length must me a multiple of the required length for sublist
    if list_len % n != 0:
        raise Exception(f"Cannot split list of {list_len} in {n} rows")

    sub_lists = list(chunks_())

    return np.array(
        [np.array(sl) for sl in sub_lists]
    )


def plot_digit(pixels: np.array, save: bool = False,
               file_name: str = "image.png"):
    """
    Plot a figure given a square matrix array,
        each cell represent a grey-scale pixel with intensity 0-1
    :param pixels: intensity of pixels
    :param save: if true, the image is stored in the directory
    :param file_name: name of file if stored (including extension)
    """

    fig, ax = plt.subplots(1)
    pixels = chunks(lst=pixels, n=SIZE)
    ax.imshow(pixels, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if save:
        file = os.path.join(IMAGES_DIR, file_name)
        logger.info(f"Saving {file}")
        create_dir(IMAGES_DIR, log=False)
        return plt.savefig(file)  # return allows inline-plot in notebooks
    else:
        plt.show()


def plot_mean_digit(X: pd.DataFrame, save: bool = False,
                    file_name: str = "image.png"):
    """
    Plots the average figure of a certain number of images
    :param X: set of images
    :param save: if true, the image is stored in the directory
    :param file_name: name of file if stored (including extension)
    """

    pixels = np.mean(X, axis=0)
    plot_digit(pixels=pixels, save=save, file_name=file_name)


def digits_histogram(labels: pd.DataFrame | np.ndarray, save: bool = False,
                     file_name: str = "plot.png"):
    """
    Plot distribution of labels in a dataset given its labels

    :param labels: collection with labels
    :param save: if true, the image is stored in the directory
    :param file_name: name of file if stored (including extension)
    """

    # type-check and casting
    if type(labels) == np.ndarray:
        labels = pd.DataFrame(labels)

    # digits count
    digits: Dict[str, int] = {
        k[0]: v for k, v in labels.value_counts().to_dict().items()
    }

    # plot
    fig, ax = plt.subplots(1)
    ax.bar(list(digits.keys()), digits.values(), edgecolor='black')
    ax.set_xticks(range(10))
    ax.set_title('Digits distribution')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Counts')

    if save:
        create_dir(IMAGES_DIR, log=False)
        return plt.savefig(os.path.join(IMAGES_DIR, file_name))
    else:
        plt.show()
