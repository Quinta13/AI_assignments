"""
This script... TODO
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from statistics import mean
from typing import Dict, Tuple, List, Iterator

from numpy import ndarray
from pandas import DataFrame

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC

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
        Predict the model
        """
        pass

    @property
    def predicted(self) -> np.ndarray:
        if self._predicted:
            return self._y_pred
        raise Exception("Classifier not predicted yet")

    @property
    def accuracy(self) -> float:
        if self._predicted:
            return accuracy_score(self._test.y, self._y_pred)
        raise Exception("Classifier not predicted yet")

    def change_dataset(self, train: Dataset, test: Dataset):
        self._train = train
        self._test = test
        self._y_pred = None
        self._fitted: bool = False
        self._predicted: bool = False

    def confusion_matrix(self):
        if self._predicted:
            cm = confusion_matrix(y_true=self._test.y, y_pred=self._y_pred)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=[str(n) for n in range(10)]
            )
            disp.plot()
        else:
            raise Exception("Classifier not predicted yet")


class SVMClassifier(Classifier):

    def __init__(self, train: Dataset, test: Dataset, c: int, degree: int):
        super().__init__(train=train, test=test)
        self.svm = SVC(C=c, degree=degree)
        self._y_pred = None

    def __str__(self):
        return super().__str__() + \
               f"C: {self.svm.C}\n" \
               f"Degree: {self.svm.degree}"

    def train(self):
        self.svm.fit(X=self._train.X, y=self._train.y)
        self._fitted = True

    def predict(self):
        self._y_pred = self.svm.predict(self._test.X)
        self._predicted = True


class SVMLinearClassifier(SVMClassifier):

    def __init__(self, train: Dataset, test: Dataset, c: int):
        super().__init__(train=train, test=test, c=c, degree=1)


class CrossValidation:

    def __init__(self, data: Dataset, k: int, classifier: Classifier):
        if k < 2:
            raise Exception("Cross validation must have at least 2 fold")
        self._data = data
        self._k = k
        self._classifier = classifier
        self._train_test = self._get_train_test()
        self._accuracy = None

    def _get_train_test(self) -> Dict[int, Tuple[Dataset, Dataset]]:
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

        accuracies = []

        # get k predicts
        for index, train_test in self._train_test.items():
            print(f"Processing fold {index + 1}")
            train, test = train_test
            self._classifier.change_dataset(
                train=train,
                test=test
            )
            self._classifier.train()
            self._classifier.predict()
            accuracies.append(self._classifier.accuracy)

        self._accuracy = mean(accuracies)

    @property
    def accuracy(self) -> float | None:
        return self._accuracy


class ClassifierTuning:

    def __init__(self, classifiers: List[Classifier],
                 data: Dataset, k: int = 10):
        self._classifiers = classifiers
        self._data = data
        self._k = k
        self._best_accuracy = 0
        self._best_classifier = None

    @property
    def best_model(self):
        return self._best_classifier

    def evaluate_best_model(self):
        for classifier in self._classifiers:
            print(f"Evaluating classifier: \n{classifier}")
            cross_validation = CrossValidation(
                data=self._data,
                k=self._k,
                classifier=classifier
            )
            cross_validation.evaluate()
            accuracy = cross_validation.accuracy
            print(accuracy)
            if accuracy > self._best_accuracy:
                print("Change")
                self._best_accuracy = accuracy
                self._best_classifier = classifier
