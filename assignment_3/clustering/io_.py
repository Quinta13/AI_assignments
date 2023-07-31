import os

from assignment_3.clustering.settings import LOG


def log(info: str):
    """
    Log message if LOG flag is enabled
    :param info: message to be logged
    """
    if LOG:
        print(info)


def makedir(path_: str):
    """
    Create a directory if it didn't exist
    :param path_: path of directory to be logged
    """
    try:
        os.makedirs(path_)
        log(f"Created directory {path_} ")
    except OSError:
        pass
