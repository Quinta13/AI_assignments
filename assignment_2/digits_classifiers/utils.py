"""
This script ...
"""

import numpy as np
import pandas as pd
from typing import List


def chunks(lst: List, n: int) -> np.array:
    """
    Split given list in a matrix with fixed-length rows length
    :param lst: list to split
    :param n: length of sublist
    :return: matrix with n rows
    """

    def chunks_():
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    list_len = len(lst)

    if list_len % n != 0:
        raise Exception(f"Cannot split list of {list_len} in {n} rows")

    sub_lists = list(chunks_())

    return np.array(
        [np.array(sl) for sl in sub_lists]
    )
