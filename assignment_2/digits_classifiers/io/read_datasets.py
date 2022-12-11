from typing import Tuple

import pandas as pd
import os.path as path

from assignment_2.digits_classifiers.model import Dataset
from assignment_2.digits_classifiers.settings import DATASETS, TRAINING_DATA, TRAINING_LABELS

in_data = path.join(DATASETS, TRAINING_DATA)
in_labels = path.join(DATASETS, TRAINING_LABELS)


def read_datasets() -> Dataset:
    """
    Read datasets
    """
    print("Reading datasets")
    return Dataset(
        x=pd.read_csv(in_data),
        y=pd.read_csv(in_labels).values.ravel()
    )
