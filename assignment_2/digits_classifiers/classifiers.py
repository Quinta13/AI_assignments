"""
This file... TODO
"""

from __future__ import annotations

from abc import ABC
from statistics import mode
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from assignment_2 import Classifier, Dataset

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


class KNNEstimator(BaseEstimator):

    def __init__(self, k: int = 1):
        """

        :param k: number of neighbor
        """

        # Training set is stored as a matrix as an array,
        #   instead of using the Dataset class
        #   to enforce performances
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.k: int = k

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Save the Training set
        :param X: feature space
        :param y: labels
        """
        self.X_train = np.array(X)
        self.y_train = y

    def _predict_one(self, x: np.ndarray) -> int:
        """
        Evaluate prediction for a single test instance
        :param x: test instance in the feature space
        :return: predicted label
        """

        train_len = self.X_train.shape[0]  # elements in the Training set

        # evaluating distances
        distances = np.array([
            np.linalg.norm(x - self.X_train[[i], :])
            for i in range(train_len)
        ])

        # index of the k smaller one
        idx = distances.argsort()[0:self.k]

        # labels of the k-nearest neighbor
        labels = self.y_train[idx]

        # majority vote
        return mode(labels)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Evaluate predictions over the test set
        :param X: feature space of Test set
        :return: prediction
        """

        X = np.array(X)  # cast to array to enforce performance
        predictions = np.array([])  # collection of y_pred

        test_len = X.shape[0]  # elements in the Training set

        for i in range(test_len):
            row = X[[i], :]  # instance of the Test set
            pred = self._predict_one(x=row)  # prediction for the instance
            predictions = np.append(predictions, pred)

        return predictions


class KNN(Classifier, ABC):
    """
    This class represent a K-nearest-neighborhood classifier
        it considers following parameters:
            - k, number of neighbors
    """

    classifier_name = f"KNN"

    def __init__(self, train: Dataset, test: Dataset,
                 params: Dict | None = None):
        """

        :param train: train dataset
        :param test: test dataset
        :param params: dictionary with possible keys 'k';
        """

        # super class
        super().__init__(train=train, test=test, params=params)

        # params
        if params is None:
            params = {}

        self.k: int = params["k"] if "k" in params.keys() else 1

        # estimator
        self._estimator: BaseEstimator = KNNEstimator(k=self.k)

    def __str__(self):
        """
        Return string representation of the object
        """
        return super(KNN, self).__str__() + f" [k: {self.k}]"

    def params(self) -> Dict[str, Any]:
        """
        :return: hyper-parameters
        """
        return {"k": self.k}

    @staticmethod
    def default_estimator() -> BaseEstimator:
        """
        :return: classifier default estimator
        """
        return KNNEstimator()


""" NAIVE BAYES """


class NaiveBayesEstimator(BaseEstimator):
    def __init__(self):
        """
        BayesEstimator have no actually hyper-parameters
        """

        # list of all possible labels
        self.labels: List[int] = []

        # dictionary which associate each label
        #  to a collection of alphas and betas, once for each pixel
        self._label_alpha_beta: Dict[int, Tuple[np.array, np.array]] | None = None

        # frequency of labels
        self._labels_frequency: Dict[int, float] | None = None


    @staticmethod
    def _get_alpha_beta(df: pd.DataFrame) -> Tuple[np.array, np.array]:
        """
        Given a data frame which represent a single class compute
            alphas and betas for the beta-distribution
        :param df: dataframe of a single class
        :return: alphas and betas for the beta distribution
        """

        # exploit of element-wise numpy operation

        mean = np.mean(df, axis=0)       # E[X]
        var = np.var(df, axis=0)         # Var[X]
        k = mean * (1 - mean) / var - 1  # K = ( E[X] * (1 - E[X]) / Var[X] ) - 1
        alpha = k * mean                 # alpha = K E[X] + 1
        beta = k * (1 - mean)            # beta  = K (1 - E[X]) + 1

        return alpha, beta

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Save the alphas and betas for each class (and for each pixel)
        Save the relative frequency of each class
        :param X: feature space
        :param y: labels
        """

        # labels
        self.labels = [int(l) for l in list(set(y))]

        # split the dataset associating to each label its rows in the dataframe
        labeled_dataset: Dict[int, pd.DataFrame] = {
            label: X.loc[y == label] for label in self.labels
        }

        # computing relative frequency of each label
        self._labels_frequency = {
            k[0]: v / len(X) for k, v in pd.DataFrame(y).value_counts().to_dict().items()
        }

        # computing alpha and beta for each label
        self._label_alpha_beta = {
            label: self._get_alpha_beta(df=df)
            for label, df in labeled_dataset.items()
        }

    def _label_product(self, label: int, x: np.ndarray) -> float:
        """
        Compute the multiplication of the betas distributions for a certain label
            using values of a certain test instance
        :param label: label
        :param x: point in the Test set
        :return: product of the beta distribution for each pixel evaluated in a certain point
        """

        alpha, beta = self._label_alpha_beta[label]
        epsilon = 0.05  # length of neighborhood

        # cumulative density function in the neighbor
        probs = beta_distribution.cdf(a=alpha, b=beta, x=x + epsilon) - \
                beta_distribution.cdf(a=alpha, b=beta, x=x - epsilon)

        # where the probability distributione doesn't exist (variance less or equal to zero)
        #   we assign one in order to not affect the multiplication
        np.nan_to_num(probs, nan=1., copy=False)

        return np.product(probs)

    def _labels_products(self, x: np.array) -> List[Tuple[float, int]]:
        """
        Compute the multiplication of beta distributions for each labels
            using value of a certain test instance
        :param x: point in the Test set
        :return: product of distribution and associated label
        """
        # List of tuple (product, label)
        # the order allows for an easy maximum search
        return [
            (
                # the product of distribution is multiplied by the probability of the class (its frequency)
                self._labels_frequency[l] * self._label_product(label=l, x=x),
                l
            )
            for l in self.labels
        ]

    def _predict_one(self, x: np.array) -> int | None:
        """
        It takes the class with the higher probability product
        :return: predicted label, None if all class have probability zero
        """
        products = self._labels_products(x=x)
        higher = max(products)
        prob, pred = higher
        if prob > 0:
            return pred
        return None  # all classes have 0 probability

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        It predict the label for all instances in the test set
        :param X: Test set
        :return: predicted labels
        """

        X = np.array(X)  # cast to array to enforce performance
        predictions = np.array([])  # collection of y_pred

        test_len = X.shape[0]  # elements in the Training set

        for i in range(test_len):
            row = X[i, :]  # instance of the Test set
            pred = self._predict_one(x=row)  # prediction for the instance
            predictions = np.append(predictions, pred)

        return predictions


class NaiveBayes(Classifier, ABC):
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
        self._estimator: BaseEstimator = NaiveBayesEstimator()

    def __str__(self):
        """
        Return string representation of the object
        """
        return super(NaiveBayes, self).__str__()

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
        return NaiveBayesEstimator()
