import pandas as pd
import numpy as np
import scipy.stats


class ChiSquareIndependentTwoVariables:
    """
    总样本量需要大于40
    理论频数低于5的单元格个数不超过20%
    任意单元格不能为0
    等级变量不可以使用
    配对变量不可以使用
    """

    def __init__(self, array_1: np.ndarray, array_2: np.ndarray):
        self.array_1 = array_1
        self.array_2 = array_2

    def test(self):
        cross_table = pd.crosstab(self.array_1, self.array_2)
        statistic, p = scipy.stats.chisquare(cross_table[0], cross_table[1])

        pass