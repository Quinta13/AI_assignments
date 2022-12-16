"""
This file... TODO
"""

from __future__ import annotations

from abc import ABC
from statistics import mode
from typing import List, Callable, Dict, Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from loguru import logger

from assignment_2.digits_classifiers.model import Classifier, Dataset
from assignment_2.digits_classifiers.utils import MinElementCollection

""" SIMPLE VECTOR MACHINE """


class SVM(Classifier, ABC):
    """
    This class represent a Support Vector Machine classifier
        it considers following parameters:
            - C, regularization parameter
            - kernel, kernel-type of the algorithm
            - degree, degree of function for polynomial kernel
    """

    classifier_name = f"SVM"

    def __init__(self, train: Dataset, test: Dataset,
                 params: Dict | None = None):
        """

        :param train: train dataset
        :param test: test dataset
        :param params: dictionary with possible keys 'C'; 'degree'; 'kernel'
        """

        # super class
        super().__init__(train=train, test=test, params=params)

        # params
        if params is None:
            params = {}

        self._C: float    = params["C"]      if "C"      in params.keys() else 1.
        self._degree: int = params["degree"] if "degree" in params.keys() else 3
        self._kernel: str = params["kernel"] if "kernel" in params.keys() else "rbf"

        # estimator
        self._estimator: BaseEstimator = SVC(C=self._C, kernel=self._kernel, degree=self._degree)

        self.classifier_name = "LinearSVM" if self._kernel == 'linear' else \
            "PolynomialSVM" if self._kernel == "poly" else "RBFKernelSVM"

    def __str__(self):
        """
        Return string representation of the object
        """
        parameters = f" [C: {self._C}; degree: {self._degree}]" if self.classifier_name == "PolynomialSVM" \
            else f"[C: {self._C}]"
        return super(SVM, self).__str__() + parameters

    def params(self) -> Dict[str, Any]:
        """
        :return: hyper-parameters
        """
        return {
            "C": self._C,
            "kernel": self._kernel,
            "degree": self._degree
        }

    @staticmethod
    def default_estimator() -> BaseEstimator:
        """
        :return: classifier default estimator
        """
        return SVC()


""" RANDOM FOREST """


class RandomForest(Classifier, ABC):
    """
    This class represent a Random Forest classifier
        it considers following parameters:
            - n_estimators, number of three of the forest
            - max_depth, maximum depth of the tree
    """

    classifier_name = f"RandomForest"

    def __init__(self, train: Dataset, test: Dataset,
                 params: Dict | None = None):
        """

        :param train: train dataset
        :param test: test dataset
        :param params: dictionary with possible keys 'n_estimators'; 'max_depth'
        """

        # super class
        super().__init__(train=train, test=test, params=params)

        # params
        if params is None:
            params = {}

        self._n_estimators: int        = params["n_estimators"] if "n_estimators" in params.keys() else 100
        self._max_depth:    int | None = params["max_depth"]    if "max_depth"    in params.keys() else None

        # estimator
        self._estimator: BaseEstimator = RandomForestClassifier(
            n_estimators=self._n_estimators, max_depth=self._max_depth, n_jobs=-1
        )

    def __str__(self):
        """
        Return string representation of the object
        """
        return super(RandomForest, self).__str__() + f" [N-trees: {self._n_estimators}; Max-depth: {self._max_depth}]"

    def params(self) -> Dict[str, Any]:
        """
        :return: hyper-parameters
        """
        return {
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth
        }

    @staticmethod
    def default_estimator() -> BaseEstimator:
        """
        :return: classifier default estimator
        """
        return RandomForestClassifier()


""" K - NEAREST NEIGHBOR """


class Neighbor:
    """
    This class represent a neighbor of an instance,
        it describes its label and the distance between the two
    """

    def __init__(self, label: int, distance: float):
        """

        :param label: label of the neighbor
        :param distance: distance between instance and neighbor
        """
        self.label: int = label
        self.distance: float = distance

    def __str__(self) -> str:
        """
        :return: string representation of the object
        """
        return f"[Label: {self.label} - dist: {self.distance}]"

    def __repr__(self) -> str:
        """
        :return: string representation of the object
        """
        return str(self)

    def __gt__(self, other: Neighbor) -> bool:
        return self.distance > other.distance

    def __lt__(self, other: Neighbor) -> bool:
        return not self.__gt__(other=other)


class Neighbourhood:
    """
    This class represent the neighborhood of an instance
    """

    def __init__(self, neighbourhood: List[Neighbor]):
        """

        :param neighbourhood: list of neighborhood
        """
        self._neighbourhood: List[Neighbor] = neighbourhood

    def __str__(self) -> str:
        """
        :return: string representation of the object
        """
        return str(self._neighbourhood)

    def __repr__(self) -> str:
        """
        :return: string representation of the object
        """
        return str(self)

    @property
    def neighbourhood(self) -> List[Neighbor]:
        """
        :return: list of neighborhoods
        """
        return self._neighbourhood

    @property
    def mode_neighbourhood(self) -> int:
        """
        :return: mode of the label of the neighbors
        """
        return mode([n.label for n in self._neighbourhood])


class KNN(Classifier, ABC):
    classifier_name = "k-NEAREST-NEIGHBORHOOD"

    def __init__(self, train: Dataset, test: Dataset,
                 distance_fun: Callable, k: int = 1):
        """

        :param train: train dataset
        :param test: test dataset
        :param distance_fun: function to compute distance between two arrays
        :param k: number of neighbors
        """
        super().__init__(train=train, test=test)
        self._distance_fun = distance_fun
        self._k = k

    def __str__(self) -> str:
        """
        Return string representation of the object
        """
        return super().__str__() + f" - [K: {self._k}]"

    def train(self):
        """
        Train the dataset
        """
        pass

    def predict(self):
        """
        Evaluate predictions over the test set
        """

        predictions = []  # list of y_pred

        for idx, test in enumerate(self._test.X.iterrows()):  # predict foreach instance in test
            _, test = test
            int(idx)
            if idx % 10 == 0:
                logger.info(f" > {idx * 100 / len(self._test):.3f}%")
            test = test.values  # test row as an array
            neighs = MinElementCollection(k=self._k)  # collection of neighbors
            for row, label in zip(self._train.X.iterrows(), self._train.y):  # iterate over train set
                _, train = row
                train = train.values  # train row as an array
                dist = self._distance_fun(test, train)
                neighs.push(Neighbor(distance=dist, label=label))
            neighborhood = Neighbourhood(neighbourhood=neighs.elements)
            predictions.append(neighborhood.mode_neighbourhood)

        self._y_pred = np.array(predictions)
        self._predicted = True
