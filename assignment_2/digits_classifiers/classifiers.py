"""
This file... TODO
"""

from __future__ import annotations

from abc import ABC
from statistics import mode
from typing import List, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from loguru import logger

from assignment_2.digits_classifiers.model import Classifier, Dataset
from assignment_2.digits_classifiers.utils import MinElementCollection

""" SIMPLE VECTOR MACHINE """


class SVM(Classifier, ABC):
    """
    This abstract class represent a Support Vector Machine classifier
    """

    classifier_name = f"SVM"

    def __init__(self, train: Dataset, test: Dataset, svm: SVC):
        """

        :param train: train dataset
        :param test: test dataset
        :param svm: support vector machine classifier
        """
        super().__init__(train=train, test=test)
        self.svm = svm

    def train(self):
        """
        Train the dataset
        """
        self.svm.fit(X=self._train.X, y=self._train.y)
        self._fitted = True

    def predict(self):
        """
        Evaluate predictions over the test set
        """
        self._y_pred = self.svm.predict(self._test.X)
        self._predicted = True


class SVMPolynomialClassifier(SVM):
    """
    This class represent a Support Vector Machine classifier with hyperparameter:
        - C: Regularization factor
        - degree: Degree of polynomial kernel function
    """

    classifier_name = f"PolynomialSVM"

    def __init__(self, train: Dataset, test: Dataset, c: float = 1., degree: int = 1):
        """

        :param train: train dataset
        :param test: test dataset
        :param c: regularization factor
        :param degree: degree of polynomial
        """
        super().__init__(train=train, test=test,
                         svm=SVC(C=c, degree=degree, kernel="poly", ))

    def __str__(self) -> str:
        """
        Return string representation of the object
        """
        return super().__str__() + f" - [C: {self.svm.C}, Degree: {self.svm.degree}]"


class SVMLinearClassifier(SVM):
    """
    This class represent a Linear Support Vector Classifier
        it's a sub-case of PolynomialSVM with degree equal to one
    """

    classifier_name = "LinearSVM"

    def __init__(self, train: Dataset, test: Dataset, c: int):
        """

        :param train: train dataset
        :param test: test dataset
        :param c: regularization factor
        """
        super().__init__(train=train, test=test, svm=SVC(C=c, kernel='linear'))

    def __str__(self) -> str:
        """
        Return string representation of the object
        """
        return super().__str__() + f" - [C: {self.svm.C}]"


class SVMRBFKernelsClassifier(SVM):
    """
    This class represent a Support Vector Classifier with a RBF kernel
    """

    classifier_name = "RBFKernelsSVM"

    def __init__(self, train: Dataset, test: Dataset, c: int):
        """

        :param train: train dataset
        :param test: test dataset
        :param c: regularization factor
        """
        super().__init__(train=train, test=test, svm=SVC(C=c, kernel="rbf"))

    def __str__(self) -> str:
        """
        Return string representation of the object
        """
        return super().__str__() + f" - [C: {self.svm.C}]"


""" RANDOM FOREST """


class RandomForest(Classifier, ABC):

    classifier_name = f"RandomForest"

    def __init__(self, train: Dataset, test: Dataset,
                 n_estimators: int = 100, max_depth: int = 100):
        """
        :param train: train dataset
        :param test: test dataset
        :param n_estimators: number of trees
        :param max_depth:
        """
        super().__init__(train=train, test=test)
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._random_forest = RandomForestClassifier(max_depth=self._max_depth, n_estimators=self._n_estimators)

    def train(self):
        """
        Train the dataset
        """
        self._random_forest.fit(X=self._train.X, y=self._train.y)
        self._fitted = True

    def predict(self):
        """
        Evaluate predictions over the test set
        """
        self._y_pred = self._random_forest.predict(self._test.X)
        self._predicted = True

    def __str__(self):
        """
        Return string representation of the object
        """
        return super().__str__() + f" - [N-estimators: {self._n_estimators}; Max-Depth: {self._max_depth}]"


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
