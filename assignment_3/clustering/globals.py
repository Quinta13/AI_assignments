""" I/O PATHS """
from os import path

from clustering.settings import DATASETS, IMAGES


def get_root_dir() -> str:
    """
    :return: path to root directory
    """
    return str(path.abspath(path.join(__file__, "../")))


def get_dataset_dir() -> str:
    """
    :return: path to dataset directory
    """
    return path.join(get_root_dir(), DATASETS)


def get_images_dir() -> str:
    """
    :return: path to images directory
    """
    return path.join(get_root_dir(), IMAGES)

