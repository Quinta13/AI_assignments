"""
This script ...
"""
from __future__ import annotations

import numpy as np
import os
import pandas as pd
from typing import List, Any

from loguru import logger
from matplotlib import pyplot as plt

from assignment_2.digits_classifiers.settings import IMAGES, get_root_dir


class MinElementCollection:
    """ This class represent a collection of at least k elements;
        it keep only the minimum k elements inserted """

    def __init__(self, k: int):
        """

        :param k: max size of the collection
        :param f: guide criteria (min or max) TODO extend to generic lamda
        """
        self._k: int = k
        self._elements: List[Any] = []

    def __iter__(self):
        return iter(self._elements)

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self._elements)

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    def push(self, element: Any):
        """
        Add en element to the collection according polices
        """
        if len(self._elements) < self._k:
            self._elements.append(element)
        else:
            max_ = max(self._elements)
            if element < max_:
                self._elements[self._elements.index(max_)] = element

    @property
    def elements(self) -> List:
        return self._elements


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
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    list_len = len(lst)

    if list_len % n != 0:
        raise Exception(f"Cannot split list of {list_len} in {n} rows")

    sub_lists = list(chunks_())

    return np.array(
        [np.array(sl) for sl in sub_lists]
    )


def plot_digit(pixels: np.array, save: bool = False,
               file_name: str = "image.png"):
    fig, ax = plt.subplots(1)
    ax.imshow(pixels, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if save:
        image_dir = os.path.join(get_root_dir(), IMAGES)
        create_dir(image_dir, log=False)
        return plt.savefig(os.path.join(image_dir, file_name))
    else:
        plt.show()


def plot_digit_distribution(labels: pd.DataFrame | np.ndarray, save: bool = False,
                            file_name: str = "plot.png"):
    if type(labels) == np.ndarray:
        labels = pd.DataFrame(labels)

    digits = {
        k[0]: v for k, v in labels.value_counts().to_dict().items()
    }

    fig, ax = plt.subplots(1)
    ax.bar(list(digits.keys()), digits.values(), edgecolor='black')
    ax.set_xticks(range(10))

    if save:
        image_dir = os.path.join(get_root_dir(), IMAGES)
        create_dir(image_dir, log=False)
        return plt.savefig(os.path.join(image_dir, file_name))
    else:
        plt.show()
