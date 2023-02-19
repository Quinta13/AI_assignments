"""
This file ... TODO
"""
from __future__ import annotations

from os import path
from typing import Iterator, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from assignment_3.clustering.settings import DATASET_DIR, IMAGES_DIR
from assignment_3.clustering.utils import digits_histogram, create_dir

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
        self.y: np.ndarray = np.array(y)

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
            x=pd.DataFrame(PCA(n_components=n_components).fit_transform(self.X)),
            y=self.y
        )

    def digit_distribution(self, save: bool = False, file_name: str = 'histo.png'):
        """
        STUB for util function
        :param save: true to save the image
        :param file_name: file name if image is saved
        """
        digits_histogram(labels=self.y, save=save, file_name=file_name)

    def reduce_to_percentage(self, percentage: float = 1.) -> Dataset:
        """
        Return a randomly reduced-percentage dataset
        return: new dataset
        """

        if not 0. <= percentage <= 1.:
            raise Exception(f"Percentage {percentage} not in range [0, 1] ")

        _, X, _, y = train_test_split(self.X, self.y, test_size=percentage)

        return Dataset(X, y)

    def store(self, X_name: str = 'dataX.csv', y_name: str = 'datay.csv'):
        """
        Stores the dataset in datasets directory
        :param X_name: name of feature file
        :param y_name: name of labels file
        """

        # Create dir if doesn't exist
        create_dir(DATASET_DIR)

        x_out = path.join(DATASET_DIR, X_name)
        y_out = path.join(DATASET_DIR, y_name)

        logger.info("Saving data")

        self.X.to_csv(x_out, index=False)
        pd.DataFrame(self.y).to_csv(y_out, index=False)

    def rescale(self) -> Dataset:
        """
        Rescales rows and columns in interval [0, 1]
        """
        self.X = pd.DataFrame(MinMaxScaler().fit_transform(self.X), columns=self.X.columns)
        return self


""" MEAN SHIFT """


class MeanShiftClustering:

    def __init__(self, data: Dataset, kernel: float):
        """

        :param data: dataset for evaluation
        :param kernel: bandwidth for mean-shift
        """

        _X, _y = data
        self._X: pd.DataFrame = _X
        self._y: np.ndarray = _y

        self._kernel: float = 0.
        self.set_kernel(kernel)

        self._trained: bool = False
        self._out: np.ndarray | None = None

        self.mean_shift = MeanShift(bandwidth=self._kernel)

    def set_kernel(self, kernel: float):
        """
        Change value of bandwidth for mean_shift,
            makes an integrity check on the value
        :param kernel: bandwidth for mean shift
        """

        if kernel <= 0:
            raise Exception(f"Given bandwidth {kernel}, but is should be positive")

        if self._kernel != kernel:
            self._trained = False
            self._kernel = kernel

    def fit(self):
        """
        Train the model
        """
        self.mean_shift = MeanShift(bandwidth=self._kernel).fit(self._X)
        self._out = self.mean_shift.labels_
        self._trained = True

    @property
    def out(self) -> np.ndarray:
        """
        Return cluster indexing after fitting
        :return: cluster indexing
        """
        self._check_trained()
        return self._out

    @property
    def n_clusters(self) -> int:
        """
        Return the number of cluster found
        :return: number of clusters found
        """
        return len(set(self.out))

    @property
    def score(self) -> float:
        """
        Return random index score
        :return: random index score
        """
        self._check_trained()
        return rand_score(self._out, self._y)

    @property
    def n_features(self):
        """
        Returns number of features used
        """
        return len(self._X.columns)

    def _check_trained(self):
        """
        Raise an exception if model was not trained yet
        """
        if not self._trained:
            raise Exception("Model not trained yet")

    def __str__(self) -> str:
        return f"[N-rows: {len(self._X)}; N-components: {self._X.shape[1]}" +\
            (f", Score: {self.score}, N-clusters: {self.n_clusters}" if self._trained else "") + "]"

    def __repr__(self) -> str:
        return str(self)


class MeanShiftEvaluation:

    COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    SCORE = 'score'
    N_CLUSTERS = 'n_clusters'

    def __init__(self, data: Dataset, n_components: List[int], kernels: List[float]):
        """

        :param data: dataset for evaluation
        :param n_components: list of number of components to evaluate
        :param kernels: list of kernel sizes
        """
        self.data: Dataset = data
        self._n_components: List[int] = n_components
        self._kernels: List[float] = kernels
        if len(self._n_components) > len(self.COLORS):
            raise Exception(f"Due to graphic representation at most {len(self.COLORS)} are considered, "
                            f"{len(self._kernels)} were given.")
        self._evaluated: bool = False
        self._best_model: MeanShiftClustering | None = None
        self._results: Dict[float, Dict[int, Dict[str, int]]] = dict()

    def __str__(self) -> str:
        return f"MeanShiftEvaluation [n_components: {self._n_components}, kernels: {self._kernels}]"

    def __repr__(self) -> str:
        return str(self)

    def _check_evaluated(self):
        if not self._evaluated:
            raise f"Evaluation not completed yet"

    def evaluate(self):
        """
        TODO
        """

        kernels = {}  # kernel size : dictionary keyed by number of components

        for k in self._kernels:
            logger.info(f"Processing kernel size: {k}")
            components = {}  # number of components : results
            for nc in self._n_components:
                data_d = self.data.make_pca(n_components=nc).rescale()
                logger.info(f"  > Processing number of component: {nc}")
                mean_shift = MeanShiftClustering(data=data_d, kernel=k)
                mean_shift.fit()
                results = {
                    self.SCORE: mean_shift.score,
                    self.N_CLUSTERS: mean_shift.n_clusters
                }
                if self._best_model is None or mean_shift.score > self._best_model.score:
                    self._best_model = mean_shift
                components[nc] = results
            kernels[k] = components

        self._evaluated = True
        self._results = kernels

    @property
    def results(self) -> Dict[float, Dict[int, Dict[str, int]]]:
        self._check_evaluated()
        return self._results

    @property
    def best_model(self) -> MeanShiftClustering:
        return self._best_model

    def _plot(self, title: str, res: str, y_label: str,
              save: bool = False, file_name: str = 'graph.png'):
        """
        TODO
        """

        for kernel, dims in self.results.items():

            c = self.COLORS[list(self.results.keys()).index(kernel)]  # get a different color for a specific kernel size
            x = []  # number of components
            y = []  # result

            for nc, out in dims.items():
                x.append(nc)
                y.append(out[res])

            # Plot the points connected by a line
            plt.plot(x, y, '-o', label=f'{kernel}  ', color=c)

        # Add a legend
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

        # Set the x and y axis labels
        plt.title(title)
        plt.xlabel('Number of components')
        plt.ylabel(y_label)

        # Show the plot
        if save:
            path.join(IMAGES_DIR, file_name)
            plt.savefig(file_name)

        plt.show()

    def plot_score(self, save=False, file_name='accuracy.png'):
        self._plot(title="Random Index Score", res=self.SCORE,
                   y_label='Score', save=save, file_name=file_name)

    def plot_n_clusters(self, save=False, file_name='n_clusters.png'):
        self._plot(title="Varying Cluster Number", res=self.N_CLUSTERS,
                   y_label='NClusters', save=save, file_name=file_name)


def split_dataset(data: Dataset, index=np.ndarray) -> Dict[Dataset, int]:
    """
    Split the Dataset in multiple given a certain index
    :param data: dataset to split
    :param index: indexes for split
    :return: dataset split according to index
    """
    values = list(set(index))  # get unique values
    return {
        v: Dataset(
            x=data.X[index == v].reset_index(drop=True),
            y=data.y[index == v]
        )
        for v in values
    }