from typing import Tuple

import pandas as pd
import os.path as path

from assignment_2.digits_classifiers.settings import DATASETS, TRAINING_DATA, TRAINING_LABELS

in_data = path.join(DATASETS, TRAINING_DATA)
in_labels = path.join(DATASETS, TRAINING_LABELS)


def read_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read datasets
    """
    print("Reading datasets")
    return pd.read_csv(in_data), pd.read_csv(in_labels)
