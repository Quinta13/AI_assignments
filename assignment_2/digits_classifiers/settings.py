import os.path as path


def get_root_dir() -> str:
    """
    Returns the path to the root of directory project.
    :return: string representing the dir path
    """

    # Remember that the relative path here is relative to __file__,
    # so an additional ".." is needed
    return str(path.abspath(path.join(__file__, "../")))


DATASETS = "datasets"
IMAGES = "images"
TRAINING_DATA = "pixels.csv"
TRAINING_LABELS = "labels.csv"
