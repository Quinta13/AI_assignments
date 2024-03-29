"""
This file... TODO
"""

from __future__ import annotations

from os import path

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from statistics import mean
from loguru import logger
from typing import Dict, Tuple, List, Iterator, Any

from numpy import ndarray
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import GridSearchCV

from assignment_2 import IMAGES, get_root_dir
from assignment_2 import digits_histogram


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

    def digit_distribution(self, save: bool = False, file_name: str = 'histo.png'):
        """
        STUB for util function
        :param save: true to save the image
        :param file_name: file name if image is saved
        """
        digits_histogram(labels=self.y, save=save, file_name=file_name)


class Classifier(ABC):

    """ Abstract class that represent a classifier with two main
        abstract methods: train and predict """

    classifier_name = "Classifier"

    def __init__(self, train: Dataset, test: Dataset, params: Dict[str, Any]):
        """

        :param train: train dataset
        :param test: test dataset
        :param params: dictionary mapping hyperparameter name to related value
        """

        # fields declaration
        self._train: Dataset | None = None
        self._test: Dataset | None = None
        self._y_pred: np.ndarray | None = None
        self._fitted: bool | None = None
        self._predicted: bool | None = None
        self._estimator: BaseEstimator | None = None  # need to be assigned in subclassed

        self.change_dataset(
            train=train,
            test=test
        )

    def __str__(self) -> str:
        """
        :return: class stringify
        """
        return f"[{self.classifier_name}: Train {len(self._train)}, Test {len(self._test)}, "\
            f"{'' if self._fitted else 'not'} fitted, {'' if self._predicted else 'not'} predicted]"

    def __repr__(self) -> str:
        """
        :return: class representation
        """
        return str(self)

    def train(self):
        """
        Train the model
        """
        self._estimator.fit(X=self._train.X, y=self._train.y)
        self._fitted = True

    def predict(self):
        """
        Evaluate prediction to save in attribute y_pred
        """
        if not self._fitted:
            raise Exception("Classifier not fitted yet")
        self._y_pred = self._estimator.predict(X=self._test.X)
        self._predicted = True

    @property
    def estimator(self) -> BaseEstimator:
        return self._estimator

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
    def _get_without_nan(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        It ignores nan results in predictions
        """
        with_nan = list(np.where(pd.isna(self.predicted))[0])
        preds = (np.delete(self.predicted, with_nan)).tolist()
        trues = (np.delete(self._test.y, with_nan)).tolist()
        return np.array(preds), np.array(trues)

    @property
    def accuracy(self) -> float:
        """
        If prediction was evaluated return the accuracy of prediction,
            otherwise it raises an exception
        :return: accuracy score of prediction
        """
        if self._predicted:
            preds, trues = self._get_without_nan
            return accuracy_score(preds, trues)
        raise Exception("Classifier not predicted yet")

    def change_dataset(self, train: Dataset, test: Dataset):
        """
        Change classifier dataset and reset flags, but leave the hyper-parameters unchanged
        :param train: train dataset
        :param test: test dataset
        """
        self._train = train
        self._fitted: bool = False
        self.change_test(test=test)

    def change_test(self, test: Dataset):
        """
        Change classifier test dataset and reset predicted flag, but leave trained model unchanged
        :param train: train dataset
        :param test: test dataset
        """
        self._test = test
        self._y_pred = None
        self._predicted: bool = False

    def confusion_matrix(self, save: bool = False, file_name: str = "confusion_matrix.png"):
        """
        If prediction was evaluated plot the confusion matrix;
            it's possible to save the plot figure
        :param save: if true the plot is saved
        :param file_name: file name for the image if saved
        """
        if self._predicted:
            preds, trues = self._get_without_nan
            cm = confusion_matrix(y_true=trues, y_pred=preds)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=[str(n) for n in range(10)]
            )
            disp.plot()
            if save:
                save_path = path.join(get_root_dir(), IMAGES, file_name)
                logger.info(f"Saving {save_path}")
                disp.figure_.savefig(save_path, dpi=300)
        else:
            raise Exception("Classifier not predicted yet")

    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """ Return hyper-parameters as a dictionary """
        pass

    @staticmethod
    @abstractmethod
    def default_estimator() -> BaseEstimator:
        """ Return base estimator with default parameters """
        pass


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
            logger.info(f" > Processing fold {index + 1}")
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

    def __init__(self, base_estimator: BaseEstimator, classifiers: List[Classifier],
                 data: Dataset, k: int = 5):
        """

        :param: candidate classifier with different hyper-parameters to tune
        :data: dataset over which make the evaluation
        """
        self._classifiers: List[Classifier] = classifiers
        self._data: Dataset = data
        self._k: int = k
        self._evaluated: bool = False
        self._base_estimator: BaseEstimator = base_estimator
        self._tuning_params: Dict[str, Any] = self._get_tuning_params()
        self._grid_search = GridSearchCV(estimator=self._base_estimator, cv=self._k,
                                         scoring='accuracy', param_grid=self._tuning_params, n_jobs=-1)

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"[Estimator: {self._base_estimator}; K: {self._k}; Params: {self._tuning_params}]"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    @property
    def best_params(self) -> Dict:
        """
        Return the best classifier, raise an exception if validation was not evaluated
        :return: best parameters evaluated by the grid-search
        """
        if self._evaluated:
            return self._grid_search.best_params_
        raise Exception("Best model tuning was not evaluated yet")

    @property
    def best_score(self) -> float:
        """
        Return the best accuracy score, raise an exception if validation was not evaluated
        :return: best score
        """
        if self._evaluated:
            return self._grid_search.best_score_
        raise Exception("Best model tuning was not evaluated yet")

    def evaluate(self):
        """
        Evaluate the grid-search
            basically is a wrapper for sklearn method
        """
        self._grid_search.fit(self._data.X, self._data.y)
        self._evaluated = True

    def _get_tuning_params(self) -> Dict[str, Any]:
        """
        It produces the list of hyper-parameters to tune
            basing on the list of classifiers
        """

        # emptiness check
        if len(self._classifiers) == 0:
            raise Exception("No classifier given")
        params = [classifier.params() for classifier in self._classifiers]
        keys = set(params[0].keys())

        # same params check
        for param in params:
            if set(set(param.keys())) != keys:
                raise Exception("Trying to tune different type of classifiers")

        return ({
            key: list(set([param[key] for param in params])) for key in keys
        })


