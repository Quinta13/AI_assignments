from sklearn.datasets import fetch_openml
import os
import os.path as path

from assignment_2.digits_classifiers.settings import DATASETS, TRAINING_DATA, TRAINING_LABELS
from assignment_2.digits_classifiers.utils import create_dir

out_data = path.join(DATASETS, TRAINING_DATA)
out_labels = path.join(DATASETS, TRAINING_LABELS)


def main():

    # Creating Directory
    create_dir(DATASETS)

    # Downloading data
    print("Fetching data")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = y.astype(int)
    X = X / 255

    # Storing data
    print("Saving data")
    X.to_csv(out_data, index=False)
    y.to_csv(out_labels, index=False)


if __name__ == "__main__":
    main()
