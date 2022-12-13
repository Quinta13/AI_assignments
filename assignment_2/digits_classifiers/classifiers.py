"""
This file... TODO
"""

from __future__ import annotations

from sklearn.svm import SVC

from assignment_2.digits_classifiers.model import Classifier, Dataset


class SVMPolynomialClassifier(Classifier):
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
        super().__init__(train=train, test=test)
        self.svm = SVC(C=c, degree=degree, kernel="poly")
        self._y_pred = None

    def __str__(self) -> str:
        """
        Return string representation of the object
        """
        return super().__str__() + f" - [C: {self.svm.C}, Degree: {self.svm.degree}]"

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


class SVMLinearClassifier(SVMPolynomialClassifier):
    """
    This class represent a Linear Support Vector Classifier
        it's a sub-case of PolynomialSVM with degree equal to one
    """

    classifier_name = "LinearSVM"

    def __init__(self, train: Dataset, test: Dataset, c: int):
        """

        :train: train dataset
        :test: test dataset
        :c: regularization factor
        """
        super().__init__(train=train, test=test, c=c, degree=1)
        self.svm = SVC(C=c, kernel="linear")

    def __str__(self) -> str:
        """
        Return string representation of the object
        """
        return super().__str__() + f" - [C: {self.svm.C}]"


class SVMRBFKernelsClassifier(SVMPolynomialClassifier):
    """
    This class represent a Support Vector Classifier with a RBF kernel
    """

    classifier_name = "RBFKernelsSVM"

    def __init__(self, train: Dataset, test: Dataset, c: int):
        """

        :train: train dataset
        :test: test dataset
        :c: regularization factor
        """
        super().__init__(train=train, test=test, c=c)
        self.svm = SVC(c=c, kernel="rbf")

    def __str__(self) -> str:
        """
        Return string representation of the object
        """
        return super().__str__() + f" - [C: {self.svm.C}]"