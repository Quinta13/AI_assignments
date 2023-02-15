"""
This file ... TODO
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.decomposition import PCA

from assignment_3.clustering.utils import digits_histogram

""" DATASET """


class Dataset:
    """
    This class represent a Dataset as a tuple:
     - feature metrix as a pandas dataframe
     - label vector as an array
    The two must have the same length
    """

    def __init__(self, x: pd.DataFrame, y: np.ndarray):
        """

        :param x: feature matrix
        :param y: label vector
        """
        if len(x) != len(y):
            raise Exception(f"X has length {len(x)}, while y has {len(y)}")
        self.X: pd.DataFrame = x
        self.y: np.ndarray = y

    def __len__(self) -> int:
        """
        :return: rows in the feature matrix
        """
        return len(self.X)

    def __iter__(self) -> Iterator[DataFrame | ndarray]:
        """
        :return: unpacked fields
        """
        return iter([self.X, self.y])

    def __str__(self) -> str:
        """
        :return: class stringify
        """
        return f"[Features: {len(self.X.columns)}; Length: {len(self)}]"

    def __repr__(self) -> str:
        """
        :return: class representation
        """
        return f"[Features: {len(self.X.columns)}; Length: {len(self)}]"

    def make_pca(self, n_components: int) -> Dataset:
        """
        Applies principle component analysis to the feature space
        :param n_components: number of components for the reduced output dataset
            an integrity check is made to check if the required number of components is feasible
        :return: dataset with reduced number of components
        """
        if n_components < 0:
            raise Exception("Number of components must be a positive number")
        actual_components = len(self.X.columns)
        if n_components >= actual_components:
            raise Exception(f"Number of components must be less than {actual_components},"
                            "the actual number of components")
        return Dataset(
            x=PCA(n_components=n_components).fit_transform(self.X),
            y=self.y
        )

    def digit_distribution(self, save: bool = False, file_name: str = 'histo.png'):
        """
        STUB for util function
        :param save: true to save the image
        :param file_name: file name if image is saved
        """
        digits_histogram(labels=self.y, save=save, file_name=file_name)


""" PCA """
