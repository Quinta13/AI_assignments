"""
This file... TODO
"""

from __future__ import annotations

from abc import ABC
from statistics import mode, mean, variance
from typing import List, Callable, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from loguru import logger
from scipy.spatial import distance

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

        self._C: float = params["C"] if "C" in params.keys() else 1.
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

        self._n_estimators: int = params["n_estimators"] if "n_estimators" in params.keys() else 100
        self._max_depth: int | None = params["max_depth"] if "max_depth" in params.keys() else None

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


class KNNEstimator(BaseEstimator):

    def __init__(self, k: int = 1, f_distance: Callable = distance.euclidean):
        """

        :param k: number of neighbor to consider
        :param f_distance: function computing distance between two point in a vector space
        """
        self.train: Dataset | None = None
        self.k: int = k
        self.f_distance: Callable = f_distance

    def fit(self, X: pd.DataFrame, y: pd):
        self.train = Dataset(x=X, y=y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Evaluate predictions over the test set
        :return: prediction
        """

        predictions = []  # list of y_pred

        log_info = 10  # logging step

        for idx, test in enumerate(X.iterrows()):  # predict foreach instance in test
            _, test = test
            int(idx)
            if idx % log_info == 0:
                logger.info(f" > {idx * 100 / len(X):.3f}%")
            test = test.values  # test row as an array
            neighs = MinElementCollection(k=self.k)  # collection of neighbors
            for row, label in zip(self.train.X.iterrows(), self.train.y):  # iterate over train set
                _, train = row
                train = train.values  # train row as an array
                dist = self.f_distance(test, train)
                neighs.push(Neighbor(distance=dist, label=label))
            neighborhood = Neighbourhood(neighbourhood=neighs.elements)
            predictions.append(neighborhood.mode_neighbourhood)

        return np.array(predictions)


class KNN(Classifier, ABC):
    """
    This class represent a K-nearest-neighborhood classifier
        it considers following parameters:
            - k, number of neighbors
            - f_distance, distance function between two points
    """

    classifier_name = f"KNN"

    def __init__(self, train: Dataset, test: Dataset,
                 params: Dict | None = None):
        """

        :param train: train dataset
        :param test: test dataset
        :param params: dictionary with possible keys 'k'; 'f_distance'
        """

        # super class
        super().__init__(train=train, test=test, params=params)

        # params
        if params is None:
            params = {}

        self.k: int = params["k"] if "k" in params.keys() else 1
        self.f_distance: Callable = params["f_distance"] if "f_distance" in params.keys() else distance.euclidean

        # estimator
        self._estimator: BaseEstimator = KNNEstimator(
            k=self.k, f_distance=self.f_distance
        )

    def __str__(self):
        """
        Return string representation of the object
        """
        return super(KNN, self).__str__() + f" [k: {self.k}; distance: {self.f_distance}]"

    def params(self) -> Dict[str, Any]:
        """
        :return: hyper-parameters
        """
        return {
            "k": self.k,
            "f_distance": self.f_distance
        }

    @staticmethod
    def default_estimator() -> BaseEstimator:
        """
        :return: classifier default estimator
        """
        return KNNEstimator()


""" BAYES """


class PixelInfo:

    """ This class represent the information of a pixel of a certain class
        with specific relation to the beta-distribution"""

    def __init__(self, mean_: float, var: float):
        """

        :param mean_: mean of the distribution
        :param var: variance of the distribution
        """
        self.mean = mean_
        self.var = var

    @property
    def k(self) -> float:
        """
        :return: K defined as ( E[X] (1 - E[X]) / Var(X) ) - 1
        """
        return self.mean * (1 - self.mean) / self.var - 1

    @property
    def k_defined(self) -> bool:
        """
        :return: is K defined for this pixel
            if not i probably derives from invalid mathematical definitions, s.a. division by zero
        """
        return not np.isnan(self.k)

    @property
    def alpha(self) -> float:
        """
        :return: alpha parameter for beta distribution defined as K E[X]
        """
        return self.k * self.mean

    @property
    def beta(self) -> float:
        """
        :return: beta parameter for beta-distribution defined as K  (1 - E[X])
        """
        return self.k * (1 - self.mean)

    @property
    def beta_distribution_fun(self) -> Callable[[float], float]:
        """
        :return: function to evaluate the beta-distribution based on class-properties alpha and beta
            if k is not defined for the class, the function in output produces the constant value 1
        """
        if self.k_defined and self.alpha > 0 and self.beta > 0:
            e = 0.1
            return lambda x: beta.cdf(x=x+e, a=self.alpha, b=self.beta) - beta.cdf(x=x-e, a=self.alpha, b=self.beta)
        return lambda x: 1.

    def beta_distribution(self, x: float) -> float:
        """
        :param x: point of evaluation
        :return: probability mass function evaluation at point x
        """
        return self.beta_distribution_fun(x)

    def __str__(self) -> str:
        """
        :return: string representation of the object
        """
        return f"Pixel[mean: {self.mean}; var: {self.var}" +\
            (f" k: {self.k}, alpha: {self.alpha}, beta: {self.beta}" if self.k_defined else f"") +\
            f"]"

    def __repr__(self) -> str:
        """
        :return: string representation of the object
        """
        return str(self)

    @staticmethod
    def get_pixel_info(X: pd.DataFrame) -> Dict[str, PixelInfo]:
        """
        :param X: dataset representing the same label
        :return: mapping between pixel (column) and its information
        """

        # mapping between column and its values
        col_values: Dict[str, np.ndarray] = {
            pixel: X.loc[: , pixel].values for pixel in X.columns
        }
        # mapping between column and its info
        pixels: Dict[str, PixelInfo] = {
            pixel: PixelInfo(mean_=mean(values), var=variance(values))
            for pixel, values in col_values.items()
        }
        return pixels


class BayesEstimator(BaseEstimator):

    def __init__(self):
        """
        BayesEstimator have no actually hyper-parameters
        """
        # pixels maps a label to a mapping between a column and its relative information
        #   label : (column : pixel_info)
        self.pixels: Dict[str, Dict[str, PixelInfo]] | None = None

    def fit(self, X: pd.DataFrame, y: pd):
        """
        :param X: feature space
        :param y: labels
        Save pixel information for each class and for each relative pixel
        """

        labeled_dataset: Dict[str, pd.DataFrame] = {
            l: X.loc[y == l] for l in list(set(y))
        }

        self.pixels = {
            label: PixelInfo.get_pixel_info(X=X)
            for label, X in labeled_dataset.items()
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        prediction = []
        for _, row in X.iterrows():
            prediction.append(
                self._single_predict(values=row.values)
            )
        return np.array(prediction)

    def _single_predict(self, values: np.ndarray) -> int:
        """
        Compute a prediction for a test-instance
        :param values: values of instance to predict (point in the feature space)
        :return: predicted class
        """

        # tuple class, log-likelihood
        log_likelihood: List[Tuple[float, int]] = [
            (
                np.product([
                    pxl.beta_distribution(val) for val, pxl in zip(values, pixels_info.values())
                ]),
                int(label)
            ) for label, pixels_info in self.pixels.items()
        ]
        _, label = max(log_likelihood)
        return label


class BayesClassifier(Classifier, ABC):
    """
    This class represent a BayesClassifier
    """

    classifier_name = f"BayesClassifier"

    def __init__(self, train: Dataset, test: Dataset):
        """

        :param train: train dataset
        :param test: test dataset
        """

        # super class, this classifier has not hyper-parameter
        super().__init__(train=train, test=test, params={})

        # estimator
        self._estimator: BaseEstimator = BayesEstimator()

    def __str__(self):
        """
        Return string representation of the object
        """
        return super(BayesClassifier, self).__str__()

    def params(self) -> Dict[str, Any]:
        """
        :return: hyper-parameters
        """
        # it has no hyper-parameter
        return {}

    @staticmethod
    def default_estimator() -> BaseEstimator:
        """
        :return: classifier default estimator
        """
        return BayesClassifier()
