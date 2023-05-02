from __future__ import annotations

import time

from sklearn.cluster import MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture

from clustering.io_ import log
from clustering.model.model import ClusteringModel, Dataset, ClusteringModelEvaluation


# MEAN SHIFT


class MeanShiftClustering(ClusteringModel):
    """
    This class provide some methods to evaluate MeanShift Clustering over a given dataset,
        in particular it automatize model fitting phase, evaluation and result analysis
    """

    REPR_NAME = "MeanShift"

    def __init__(self, data: Dataset, kernel: float):
        """

        :param data: dataset for evaluation
        :param kernel: bandwidth for mean-shift
        """

        super().__init__(data=data)

        self._kernel: float = kernel

        self.model = MeanShift(bandwidth=self._kernel)

    def fit(self):
        """
        Train the model
        """
        self.model = MeanShift(bandwidth=self._kernel).fit(self._X)
        self._out = self.model.labels_
        self._trained = True

    def __str__(self) -> str:
        """
        Return a string representation for the class
        :return: stringify MeanShiftClustering
        """
        return super().__str__() + f"[KernelSize: {self._kernel}] "


class MeanShiftEvaluation(ClusteringModelEvaluation):
    """
    This class automatize different MeanShiftCluster models evaluation over a different combination of:
        - kernel size
        - number of components
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """
    EVALUATION_NAME = "MeanShiftEvaluation"

    HYPERPARAMETER = "kernel-size"

    def evaluate(self):
        """
        Evaluate MeanShift Clustering over all combination of
            - number of components used
            - kernel dimension (bandwidth)
        """

        self._evaluate(cm=MeanShiftClustering)


# NORMALIZED CUT


class NormalizedCutClustering(ClusteringModel):
    """
    This class provide some methods to evaluate Normalized Cut Clustering over a given dataset,
        in particular it automatize model fitting phase, evaluation and result analysis
    """

    REPR_NAME = "NormalizedCut"

    def __init__(self, data: Dataset, k: int):
        """

        :param data: dataset for evaluation
        :param k: number of clusters
        """

        super().__init__(data=data)

        self._k: float = k

        self.model = SpectralClustering(n_clusters=self._k, eigen_solver='arpack',
                                        assign_labels='kmeans', random_state=0)

    def fit(self):
        """
        Train the model
        """
        self.model.fit(self._X)
        self._out = self.model.labels_
        self._trained = True

    def __str__(self) -> str:
        """
        Return a string representation for the class
        :return: stringify NormalizedCutClustering
        """
        return super().__str__() + f"[K: {self._k}] "


class NormalizedCutEvaluation(ClusteringModelEvaluation):
    """
    This class automatize different NormalizedCutClustering models evaluation over a different combination of:
        - k (number of clusters)
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """

    EVALUATION_NAME = "NormalizedCutEvaluation"

    HYPERPARAMETER = "k"

    def evaluate(self):
        """
        Evaluate NormalizedCut Clustering over all combination of
            - number of components used
            - k (number of clusters)
        """

        self._evaluate(cm=NormalizedCutClustering)


# MIXTURE GAUSSIAN

class MixtureGaussianClustering(ClusteringModel):
    """
    This class provide some methods to evaluate Mixture Gaussian Clustering over a given dataset,
        in particular it automatize model fitting phase, evaluation and result analysis
    """

    REPR_NAME = "MixtureGaussian"

    def __init__(self, data: Dataset, k: int):
        """

        :param data: dataset for evaluation
        :param k: number of clusters
        """

        super().__init__(data=data)

        self._k: float = k

        self.model = GaussianMixture(n_components=self._k, covariance_type='full',
                                     init_params='kmeans', random_state=0)

    def fit(self):
        """
        Train the model
        """
        self._out = self.model.fit_predict(self._X)
        self._trained = True

    def __str__(self) -> str:
        """
        Return a string representation for the class
        :return: stringify MixtureGaussianClustering
        """
        return super().__str__() + f"[K: {self._k}] "


class MixtureGaussianEvaluation(ClusteringModelEvaluation):
    """
    This class automatize different NormalizedCutClustering models evaluation over a different combination of:
        - k (number of clusters)
    Provide some methods for analyzing evaluation results, such as getting the best model or plotting some trends
    """

    EVALUATION_NAME = "NormalizedCutEvaluation"

    HYPERPARAMETER = "k"

    def evaluate(self, log_: bool = True):
        """
        Evaluate MixtureGaussian Clustering over all combination of
            - number of components used
            - k (number of clusters)
        """

        self._evaluate(cm=MixtureGaussianClustering)
