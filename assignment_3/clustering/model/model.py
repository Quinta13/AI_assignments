"""

This module provides basic classes to implement clustering logic
 - Dataset, a class to provide a unique view for data and labels
 - DataClusterSplit, a class to split data into clusters
 - ClusteringModel, an abstract class to implement a clustering algorithm
 - ClusteringModelEvaluation, an abstract class to implement multiple evaluation
    over multiple hyper-parameters and dimensionality

"""

from __future__ import annotations

import time
from abc import abstractmethod, ABC
from os import path
from typing import Iterator, Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from statistics import mean, mode
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from clustering.globals import get_dataset_dir, get_images_dir
from clustering.io_ import makedir
from clustering.settings import IMG_EXT
from clustering.utils import log, digits_histogram, plot_digit, plot_mean_digit, plot_cluster_frequencies_histo

""" DATASET """


class Dataset:
    """
    This class represent a Dataset as a tuple:
     - feature metrix as a pandas dataframe
     - label vector as an array
    """

    def __init__(self, x: pd.DataFrame, y: np.ndarray):
        """
        :param x: feature matrix
        :param y: label vector
        """

        # data and labels must have the same length
        if len(x) != len(y):
            raise Exception(f"X has length {len(x)}, while y has {len(y)}")

        self._X: pd.DataFrame = x
        self._y: np.ndarray = np.array(y)

    @property
    def X(self) -> pd.DataFrame:
        """
        :return: feature matrix
        """
        return self._X

    @property
    def y(self) -> np.ndarray:
        """
        :return: labels
        """
        return self._y

    @property
    def features(self) -> List[str]:
        """
        :return: features name
        """
        return list(self.X.columns)

    # DUNDER METHODS

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
        return f"[Length: {len(self)}; Features: {len(self.X.columns)}]"

    def __repr__(self) -> str:
        """
        :return: class representation
        """
        return str(self)

    # CHANGES TO FEATURE SPACE

    def make_pca(self, n_components: int) -> Dataset:
        """
        Applies principle component analysis to the feature space
        :param n_components: number of components for the reduced output dataset
            an integrity check is made to check if the required number of components is feasible
        :return: dataset with reduced number of components
        """

        # integrity checks
        if n_components < 0:
            raise Exception("Number of components must be a positive number")

        actual_components = len(self.X.columns)

        if n_components >= actual_components:
            raise Exception(f"Number of components must be less than {actual_components},"
                            "the actual number of components")

        # return new object
        return Dataset(
            x=pd.DataFrame(PCA(n_components=n_components).fit_transform(self.X)),
            y=self.y
        )

    def reduce_to_percentage(self, percentage: float = 1.) -> Dataset:
        """
        Return a randomly reduced-percentage dataset
        return: new dataset
        """

        if not 0. <= percentage <= 1.:
            raise Exception(f"Percentage {percentage} not in range [0, 1] ")

        _, X, _, y = train_test_split(self.X, self.y, test_size=percentage)

        return Dataset(X, y)

    def rescale(self) -> Dataset:
        """
        Rescales rows and columns in interval [0, 1]
        """
        new_X = pd.DataFrame(MinMaxScaler().fit_transform(self.X), columns=self.features)
        return Dataset(
            x=new_X,
            y=self.y
        )

    # STORE

    def store(self, x_name: str = 'dataX', y_name: str = 'datay'):
        """
        Stores the dataset in datasets directory
        :param x_name: name of feature file
        :param y_name: name of labels file
        """

        makedir(get_dataset_dir())

        x_out = path.join(get_dataset_dir(), f"{x_name}.csv")
        y_out = path.join(get_dataset_dir(), f"{y_name}.csv")

        log(f"Saving {x_out}")
        self.X.to_csv(x_out, index=False)

        log(f"Saving {y_out}")
        pd.DataFrame(self.y).to_csv(y_out, index=False)

    def plot_digit_distribution(self, save: bool = False, file_name: str = 'histo.png'):
        """
        STUB for util function
        :param save: true to save the image
        :param file_name: file name if image is saved
        """
        digits_histogram(labels=self.y, save=save, file_name=file_name)

    def plot_digits(self):
        """
        Plots all digits in the dataset
        """
        for i in range(len(self)):
            pixels = np.array(self.X.iloc[i])
            plot_digit(pixels=pixels)

    def plot_mean_digit(self):
        """
        Plots mean of all digits in the dataset
        """
        plot_mean_digit(X=self.X)


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

        # materialized view of cluster actual_label - true_label for score evaluation
        self._cluster_idx, self._true_label = self._materialize_indexes()

    # DUNDER

    def __str__(self) -> str:
        """
        Return string representation for the object:
        :return: stringify Data Clustering Split
        """
        return f"ClusterDataSplit [Data: {self.total_instances}, Clusters: {self.n_cluster}, " \
               f"Mean-per-Cluster: {self.mean_cardinality:.3f}, Score: {self.rand_index_score:.3f}] "

    def __repr__(self) -> str:
        """
        Return string representation for the object:
        :return: stringify Data Clustering Split
        """
        return str(self)

    # STATS

    @property
    def clusters(self) -> Dict[int, Dataset]:
        """
        Return dataset split by clusters in format cluster_id : cluster data
        :return: data split in cluster
        """
        return self._clusters

    @property
    def n_cluster(self) -> int:
        """
        Returns the number of clusters found
        :return: number of clusters
        """
        return len(self.clusters)

    @property
    def clusters_cardinality(self) -> Dict[int, int]:
        """
        Return the number of objects for each cluster
        :return: mapping cluster_id : number of elements
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

    # ALTER THE CLUSTER

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

        dcs = DataClusterSplit(  # generating new fake DataClusterSplit
            data=Dataset(x=pd.DataFrame([0]), y=np.array([0])),
            index=np.array([0])
        )
        dcs._clusters = {  # setting new datas satisfying length bounds
            k: v for k, v in self.clusters.items()
            if a <= len(v) <= b
        }
        dcs._cluster_idx, dcs._true_label = dcs._materialize_indexes()
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

    # PLOTS

    def plot_frequencies_histo(self, save: bool = False, file_name: str = 'frequencies'):
        """
        Plot frequencies in a histogram
        :save: if to save the graph to images directory
        :file_name: name of stored file
        """

        plot_cluster_frequencies_histo(frequencies=self.clusters_frequencies, save=save, file_name=file_name)

    def plot_mean_digit(self):
        """
        Plots mean digit foreach cluster
        """

        for c in self.clusters.values():
            freq = {x: list(c.y).count(x) for x in c.y}
            freq = dict(sorted(freq.items(), key=lambda x: -x[1]))  # sort by values
            print(f"[Mode {mode(c.y)}: {freq}] ")
            plot_mean_digit(X=c.X)

    # SCORE

    def _materialize_indexes(self) -> Tuple[List[int], List[int]]:
        """
        Provides list of clusters and corresponding labels to evaluate scores
        """

        cluster_idx = [item for sublist in [
            [idx] * len(data) for idx, data in self.clusters.items()
        ] for item in sublist]

        true_labels = np.concatenate([
            data.y for _, data in self.clusters.items()
        ]).ravel().tolist()

        return cluster_idx, true_labels

    @property
    def rand_index_score(self) -> float:
        """
        :return: clustering rand index score
        """
        return rand_score(labels_true=self._true_label, labels_pred=self._cluster_idx)


""" CLUSTERING MODEL """


class ClusteringModel(ABC):
    """
    This is an abstract class for Clustering Model implementation
    """

    REPR_NAME: str = "ClusteringModel"

    def __init__(self, data: Dataset):
        """
        The initializer store dataset for evaluation
        :param data: dataset for evaluation
        """

        _X, _y = data
        self._X: pd.DataFrame = _X
        self._y: np.ndarray = _y

        self._trained: bool = False
        self._out: np.ndarray | None = None

        self.model = None

    @abstractmethod
    def fit(self):
        """ Todo provide out and save trained"""
        """ Fit the model """
        pass

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
        :return: stringify clustering model
        """
        return f"{self.REPR_NAME}[N-rows: {len(self._X)}; N-components: {self._X.shape[1]}; " + \
               (f" Score: {self.score}, N-clusters: {self.n_clusters}" if self._trained else "") + "] "

    def __repr__(self) -> str:
        """
        Return a string representation for the class
        :return: stringify clustering model
        """
        return str(self)


class ClusteringModelEvaluation(ABC):
    """
    This class automatize different clustering models evaluation over a different combination of:
        - hyperparameter (size of kernel, number of clusters)
        - number of components
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """

    # class name
    EVALUATION_NAME = 'ClusteringModelEvaluation'

    # hyperparameter
    HYPERPARAMETER = 'hyperparameter'

    # keys for result dictionary
    SCORE = 'score'
    N_CLUSTERS = 'n_clusters'
    TIME = 'time'

    def __init__(self, data: Dataset, n_components: List[int], hyperparameter: List[int | float], log_: bool = False):
        """

        :param data: dataset for evaluation
        :param n_components: list of number of components to evaluate
        :param hyperparameter: list of model hyper-parameters
        :param log_: if to log progress when evaluating
        """

        self.data: Dataset = data
        self._n_components: List[int] = n_components
        self._hyperparameters: List[int | float] = hyperparameter

        self._evaluated: bool = False
        self._best_model: ClusteringModel | None = None
        self._results: Dict[int | float, Dict[int, Dict[str, int]]] = dict()

        self._log = print if log_ else lambda x: None

    def __str__(self) -> str:
        """
        Returns string representation for the object
        :return: stringify MeanShiftClustering
        """
        return f"{self.EVALUATION_NAME} [n_components: {self._n_components}, {self.HYPERPARAMETER}: {self._hyperparameters}, " \
               f"{'not' if not self._evaluated else ''} evaluated]"

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

    @abstractmethod
    def evaluate(self):
        """TODO provide results and check evalutaed and best model"""
        pass

    def _evaluate(self, cm: type):
        """
        Evaluate a ClusteringModel over all combination of
            - number of components used
            - hyper-parameter
        Results are organized in a dictionary providing:
            - number of clusters found
            - random index score of any model
            - evaluation time
        :param cm: implementation of a specific clustering model
        """

        kernels = {}  # kernel size : dictionary keyed by number of components

        for k in self._hyperparameters:
            self._log(f"Processing {self.HYPERPARAMETER}: {k}")
            components = {}  # number of components : results
            for nc in self._n_components:
                data_d = self.data.make_pca(n_components=nc).rescale()
                cm_obj = cm(data_d, k)
                t1 = time.perf_counter()
                cm_obj.fit()
                t2 = time.perf_counter()
                elapsed = t2 - t1
                results = {
                    self.SCORE: cm_obj.score,
                    self.N_CLUSTERS: cm_obj.n_clusters,
                    self.TIME: elapsed
                }
                self._log(f"  > Processed number of component: {nc} [{elapsed:.5f} s] ")
                if self._best_model is None or cm_obj.score > self._best_model.score:
                    self._best_model = cm_obj
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
    def best_model(self) -> ClusteringModel:
        """
        Returns best model in the evaluation

        """
        self._check_evaluated()
        return self._best_model

    def _plot(self, title: str, res: str, y_label: str,
              save: bool = False, file_name: str = 'graph'):
        """
        Plot a graph foreach different kernel used:
            - x axes: number of component
            - y axes: stats (number of clusters / score / time)
        :param title: graph title
        :param res: weather score or number of cluster or time
        :param y_label: name for ordinates axes
        :param save: if to save the graph to images directory
        :param file_name: name of stored file
        """

        for kernel, dims in self.results.items():

            x = []  # number of components
            y = []  # result

            for nc, out in dims.items():
                x.append(nc)
                y.append(out[res])

            # Plot the points connected by a line
            plt.plot(x, y, '-o', label=f'{kernel}  ')

        # Add a legend
        plt.legend(bbox_to_anchor=(1, 1), title=self.HYPERPARAMETER, loc='upper left', borderaxespad=0.)

        # Set the x and y axis labels
        plt.title(title)
        plt.xlabel('Number of components')
        plt.ylabel(y_label)

        # Show the plot
        if save:
            makedir(get_images_dir())
            file_name = path.join(get_images_dir(), f"{file_name}.{IMG_EXT}")
            plt.savefig(file_name)

        # SAve the plot
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

