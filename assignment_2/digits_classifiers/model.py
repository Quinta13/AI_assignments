"""
This script... TODO
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from statistics import mean
from loguru import logger
from typing import Dict, Tuple, List, Iterator

from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from assignment_2.digits_classifiers.utils import plot_digit_distribution


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

    def digit_distribution(self):
        """
        STUB for util function
        """
        plot_digit_distribution(labels=self.y)


class Classifier(ABC):

    """ Abstract class that represent a classifier with two main
        abstract methods: train and predict """

    classifier_name = "Classifier"

    def __init__(self, train: Dataset, test: Dataset):
        """

        :param train: train dataset
        :param test: test dataset
        """

        # fields declaration
        self._train: Dataset | None = None
        self._test: Dataset | None = None
        self._y_pred: np.ndarray | None = None
        self._fitted: bool | None = None
        self._predicted: bool | None = None

        self.change_dataset(
            train=train,
            test=test
        )

    def __str__(self) -> str:
        """
        :return: class stringify
        """
        return f"[{self.classifier_name}: Train {len(self._train)}, Test {len(self._test)}]"

    def __repr__(self) -> str:
        """
        :return: class representation
        """
        return str(self)

    @abstractmethod
    def train(self):
        """
        Train the model
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Evaluate prediction to save in attribute y_pred
        """
        pass

    @property
    def predicted(self) -> np.ndarray:
        """
        If prediction was evaluated return predicted values,
            otherwise it raises an exception
        :return: predicted values
        """
        if self._predicted:
            return self._y_pred
        raise Exception("Classifier not predicted yet")

    @property
    def accuracy(self) -> float:
        """
        If prediction was evaluated return the accuracy of prediction,
            otherwise it raises an exception
        :return: accuracy score of prediction
        """
        if self._predicted:
            return accuracy_score(self._test.y, self._y_pred)
        raise Exception("Classifier not predicted yet")

    def change_dataset(self, train: Dataset, test: Dataset):
        """
        Change classifier dataset and reset flags, but leave the hyper-parameters unchanged
        :param train: train dataset
        :param test: test dataset
        """
        self._train = train
        self._test = test
        self._y_pred = None
        self._fitted: bool = False
        self._predicted: bool = False

    def confusion_matrix(self, save: bool = False, file_name: str = "confusion_matrix.png"):
        """
        If prediction was evaluated plot the confusion matrix;
            it's possible to save the plot figure
        :param save: if true the plot is saved
        :param file_name: file name for the image if saved
        """
        if self._predicted:
            cm = confusion_matrix(y_true=self._test.y, y_pred=self._y_pred)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=[str(n) for n in range(10)]
            )
            disp.plot()
            if save:
                disp.figure_.savefig(file_name, dpi=300)
        else:
            raise Exception("Classifier not predicted yet")


class KFoldCrossValidation:
    """
    This class implement a k-fold Cross Validation for a classifier over a certain dataset
    It should be used to validate some hyper-parameters
    """

    def __init__(self,  classifier: Classifier, data: Dataset, k: int = 5):
        """

        :param classifier: classifier to validate
        :param data: dataset
        :param k: number of folds
        """
        if k < 2:
            raise Exception("Cross validation must have at least 2 fold")
        self._data: Dataset = data
        self._k: bool = k
        self._classifier: Classifier = classifier
        self._train_test: Dict[int, Tuple[Dataset, Dataset]] = self._get_train_test()
        self._evaluated: bool = False
        self._accuracy: float = 0.

    def _get_train_test(self) -> Dict[int, Tuple[Dataset, Dataset]]:
        """
        It creates k sets of couples (train, test) by choosing at once
        a single fold at time as test and the rest of folds as train set
        :return: dictionary which values are the index of folds used as test and values
            are the tuple of associated train and test set
        """

        # compute step length and divide intervals
        step = round((len(self._data) + 1) / self._k)
        intervals: List[Tuple[int, int]] = [(i * step, (1 + i) * step - 1) for i in range(self._k)]
        intervals[-1] = (intervals[-1][0], len(self._data) - 1)  # adjust end of last interval

        # create folds
        folds: Dict[int, Dataset] = {
            i: Dataset(
                x=self._data.X.iloc[list(range(intervals[i][0], intervals[i][1] + 1))],
                y=self._data.y.take(list(range(intervals[i][0], intervals[i][1] + 1)))
            )
            for i in range(self._k)
        }

        # divide per train test
        train_test: Dict[int, Tuple[Dataset, Dataset]] = {
            i: (
                Dataset(
                    x=pd.concat([folds[j].X for j in range(self._k) if j != i]),
                    y=np.concatenate([folds[j].y for j in range(self._k) if j != i])
                ),
                folds[i]
            ) for i in range(self._k)
        }

        return train_test

    def evaluate(self):
        """
        It evaluates the accuracy associated to each k combination of the dataset
            and then compute the average of the accuracy of predictions
        """

        accuracies = []

        # get k predicts
        for index, train_test in self._train_test.items():
            logging.info(f" > Processing fold {index + 1}")
            train, test = train_test
            self._classifier.change_dataset(
                train=train,
                test=test
            )
            self._classifier.train()
            self._classifier.predict()
            accuracies.append(self._classifier.accuracy)

        self._accuracy = mean(accuracies)
        self._evaluated = True

    @property
    def accuracy(self) -> float:
        if self._evaluated:
            return self._accuracy
        raise Exception(f"{self._k}-fold cross validation not evaluated yet")


class ClassifierTuning:
    """ Class that allows to tune hyper-parameters
        evaluating the k-fold-cross-validation over a set of candidate classifier """

    def __init__(self, classifiers: List[Classifier],
                 data: Dataset, k: int = 5):
        """

        :param: candidate classifier with different hyper-parameters to tune
        :data: dataset over which make the evaluation
        """
        self._classifiers: List[Classifier] = classifiers
        self._data: Dataset = data
        self._k: int = k
        self._best_accuracy: float = 0
        self._evaluated: bool = False
        self._best_classifier: Classifier | None = None

    @property
    def best_model(self) -> Classifier:
        """
        Return the best classifier if accuracy was evaluated for each candidate
            otherwise it raises an exception
        :return: best classifier
        """
        if self._evaluated:
            return self._best_classifier
        raise Exception("Best model tuning was not evaluated yet")

    def evaluate_best_model(self):
        """
        Evaluate the candidate classifiers with k-fold cross validation and find the best one
        """
        for classifier in self._classifiers:
            logger.info(f"Evaluating classifier: {classifier}")

            # compute accuracy with cross validation
            cross_validation = KFoldCrossValidation(
                data=self._data,
                k=self._k,
                classifier=classifier
            )
            cross_validation.evaluate()
            accuracy = cross_validation.accuracy

            # update best one
            if accuracy > self._best_accuracy:
                self._best_accuracy = accuracy
                self._best_classifier = classifier

        self._evaluated = True
