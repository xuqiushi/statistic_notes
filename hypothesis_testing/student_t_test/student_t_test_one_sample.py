import scipy.stats
import numpy as np


class StudentTTestOneSample:
    """
    1. 方差未知
    2. 正态或近似正态
    """
    def __init__(self, array: np.ndarray):
        self.array = array
