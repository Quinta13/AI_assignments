"""
This file ... TODO
"""
from __future__ import annotations

import time
from os import path
from typing import Iterator, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray
from statistics import mean, mode
from pandas import DataFrame
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from assignment_3.clustering.settings import DATASET_DIR, IMAGES_DIR
from assignment_3.clustering.utils import digits_histogram, create_dir, plot_mean_digit

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
    """
    This class provide some methods to evaluate MeanShift Clustering over a given dataset,
        in particular it automatize model fitting phase, evaluation and result analysis
    """

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
        Change value of kernel size for mean_shift,
            makes an integrity check on the value
            if the kernel size changes the model need to be reevaluated
        :param kernel: kernels size for mean shift
        """

        if kernel <= 0:
            raise Exception(f"Given kernel size of {kernel}, but is should be positive")

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
        """
        Return a string representation for the class
        :return: stringify MeanShiftClustering
        """
        return f"[N-rows: {len(self._X)}; N-components: {self._X.shape[1]}; KernelSize: {self._kernel}" + \
               (f", Score: {self.score}, N-clusters: {self.n_clusters}" if self._trained else "") + "]"

    def __repr__(self) -> str:
        """
        Return a string representation for the class
        :return: stringify MeanShiftClustering
        """
        return str(self)


class MeanShiftEvaluation:
    """
    This class automatize different MeanShiftCluster models evaluation over a different combination of:
        - kernel size
        - number of components
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """

    COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '#E15E3F', '#6a329f']  # plotting colors depending on bandwidth

    # keys for result dictionary
    SCORE = 'score'
    N_CLUSTERS = 'n_clusters'
    TIME = 'time'

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
            raise Exception(f"Due to graphic representation at most {len(self.COLORS)} kernels sizes are considered, "
                            f"{len(self._kernels)} were given.")
        self._evaluated: bool = False
        self._best_model: MeanShiftClustering | None = None
        self._results: Dict[float, Dict[int, Dict[str, int]]] = dict()

    def __str__(self) -> str:
        """
        Returns string representation for the object
        :return: stringify MeanShiftClustering
        """
        return f"MeanShiftEvaluation [n_components: {self._n_components}, kernels: {self._kernels}]"

    def __repr__(self) -> str:
        """
        Returns string representation for the object
        :return: stringify MeanShiftClustering
        """
        return str(self)

    def _check_evaluated(self):
        """
        Check if model was evaluated,
            it raise an exception if it hasn't
        """
        if not self._evaluated:
            raise f"Evaluation not completed yet"

    def evaluate(self, log: bool = True):
        """
        Evaluate MeanShift Clustering over all combination of
            - number of components used
            - kernel dimension (bandwidth)
        Results are organized in a dictionary providing:
            - number of clusters found
            - random index score of any model
            - evaluation time
        :param log: if to log progress
        """

        log_ = print if log else lambda x: None

        kernels = {}  # kernel size : dictionary keyed by number of components

        for k in self._kernels:
            log_(f"Processing kernel size: {k}")
            components = {}  # number of components : results
            for nc in self._n_components:
                data_d = self.data.make_pca(n_components=nc).rescale()
                mean_shift = MeanShiftClustering(data=data_d, kernel=k)
                t1 = time.perf_counter()
                mean_shift.fit()
                elapsed = time.perf_counter() - t1
                results = {
                    self.SCORE: mean_shift.score,
                    self.N_CLUSTERS: mean_shift.n_clusters,
                    self.TIME: elapsed
                }
                log_(f"  > Processed number of component: {nc} [{elapsed:.5f} s] ")
                if self._best_model is None or mean_shift.score > self._best_model.score:
                    self._best_model = mean_shift
                components[nc] = results
            kernels[k] = components

        self._evaluated = True
        self._results = kernels

    @property
    def results(self) -> Dict[float, Dict[int, Dict[str, int]]]:
        """
        Provides results of evaluation in a dictionary format ( kernel size : number of components : clusters, score )
        """
        self._check_evaluated()
        return self._results

    @property
    def best_model(self) -> MeanShiftClustering:
        """
        Returns best model in the evaluation

        """
        return self._best_model

    def _plot(self, title: str, res: str, y_label: str,
              save: bool = False, file_name: str = 'graph.png'):
        """
        Plot a graph foreach different kernel used:
            - x axes: number of component
            - y axes: stats (number of clusters / score)
        :param title: graph title
        :param res: weather score or number of cluster key
        :y_label: name for ordinates axes
        :save: if to save the graph to images directory
        :file_name: name of stored file
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
        """
        Plot score graph
        :save: if to save the graph to images directory
        :file_name: name of stored file
        """
        self._plot(title="Random Index Score", res=self.SCORE,
                   y_label='Score', save=save, file_name=file_name)

    def plot_n_clusters(self, save=False, file_name='n_clusters.png'):
        """
        Plot n_cluster graph
        :save: if to save the graph to images directory
        :file_name: name of stored file
        """
        self._plot(title="Varying Cluster Number", res=self.N_CLUSTERS,
                   y_label='NClusters', save=save, file_name=file_name)

    def plot_time(self, save=False, file_name='time.png'):
        """
        Plot time execution graph
        :save: if to save the graph to images directory
        :file_name: name of stored file
        """
        self._plot(title="Elapsed Execution Time", res=self.TIME,
                   y_label='Time', save=save, file_name=file_name)


# CLUSTER DATA SPLIT

class DataClusterSplit:
    """
    Provide an interface to split a dataset given clustering index
    """

    def __init__(self, data: Dataset, index: np.ndarray):
        """

        :param data: dataset to be split
        :param index: clustering index
        """
        self._clusters: Dict[int, Dataset] = self._split_dataset(data=data, index=index)

    # STATS

    @property
    def clusters(self) -> Dict[int, Dataset]:
        """
        Return dataset split by clusters in format cluster_id : cluster data
        :return: data split by cluster
        """
        return self._clusters

    @property
    def n_cluster(self) -> int:
        """
        Returns the number of clusters found
        :return: number of clusters
        """
        return len(self._clusters)

    @property
    def clusters_cardinality(self) -> Dict[int, int]:
        """
        Return the number of points for each cluster
        :return: number of instances for each cluster
        """
        return {k: len(v) for k, v in self.clusters.items()}

    @property
    def total_instances(self) -> int:
        """
        Returns the total number of points among all clusters
        :return: total number of instances among all clusters
        """
        return sum(self.clusters_cardinality.values())

    @property
    def clusters_frequencies(self) -> Dict[int, int]:
        """
        Return the frequencies of cluster cardinality
        :return: cluster cardinality frequencies
        """
        lengths = list(self.clusters_cardinality.values())
        return {x: lengths.count(x) for x in lengths}

    @property
    def mean_cardinality(self) -> float:
        """
        Return average cluster cardinality
        :return: average cluster cardinality
        """
        return mean(self.clusters_cardinality.values())

    def __str__(self) -> str:
        """
        Return string representation for the object:
        :return: stringify Data Clustering Split
        """
        return f"Cluster Data Split [Data: {self.total_instances}, Clusters: {self.n_cluster}, Mean-per-Cluster: {self.mean_cardinality}] "

    def __repr__(self) -> str:
        """
        Return string representation for the object:
        :return: stringify Data Clustering Split
        """
        return str(self)

    def get_sub_clusters(self, a: int | None = None, b: int | None = None) -> DataClusterSplit:
        """
        Returns a new DataClusterSplit with cluster cardinalities in given range [a, b]
        :param a: cardinality lower bound, zero if not given
        :param b: cardinality upper bound, maximum cardinality if not given
        """
        if a is None:  # lower-bound to zero
            a = 0
        if b is None:  # upper-bound to maximum cardinality
            b = max(self.clusters_cardinality.values())
        dcs = DataClusterSplit(  # generating new empty DataClusterSplit
            data=Dataset(x=pd.DataFrame(), y=np.array([])),
            index=np.array([])
        )
        dcs._clusters = {  # setting new datas satisfying length bounds
            k: v for k, v in self.clusters.items()
            if a <= len(v) <= b
        }
        return dcs

    @staticmethod
    def _split_dataset(data: Dataset, index: np.ndarray) -> Dict[int, Dataset]:
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

    def frequencies_histo(self, save: bool = False, file_name: str = 'frequencies.png'):
        """
        Plot frequencies in a histogram
        :save: if to save the graph to images directory
        :file_name: name of stored file
        """

        fig, ax = plt.subplots(1)

        ax.bar(list(self.clusters_frequencies.keys()), self.clusters_frequencies.values(), edgecolor='black')

        # Title and axes
        ax.set_title('Clusters cardinality')
        ax.set_xlabel('Cluster dimension')
        ax.set_ylabel('Occurrences')

        if save:
            create_dir(IMAGES_DIR, log=False)
            return plt.savefig(path.join(IMAGES_DIR, file_name))
        else:
            plt.show()

    def plot_mean_digit(self):
        """
        Plots mean digit foreach cluster
        """

        for c in self.clusters.values():
            freq = {x: list(c.y).count(x) for x in c.y}
            print(f"[Mode {mode(c.y)}: {freq}] ")
            plot_mean_digit(X=c.X)
