import logging
from typing import Optional

import pandas as pd
import numpy as np
import scipy.stats

from hypothesis_testing.chi_square_test.entity.chi_square_result import ChiSquareResult


class ChiSquareIndependentTwoVariables:
    """
    总样本量需要大于40
    理论频数低于5的单元格个数不超过20%
    任意单元格不能为0
    等级变量不可以使用
    配对变量不可以使用
    """

    def __init__(
        self, array_1: np.ndarray, array_2: np.ndarray, p_thread: float = 0.05
    ):
        self.array_1 = array_1  # 代表分类
        self.array_2 = array_2  # 代表每个分类下的不同效果
        self.p_thread = p_thread
        logging.basicConfig(level=logging.INFO)
        logging.info(f"{'ChiSquareIndependentTwoVariables':=^50}")
        logging.info(f"卡方检验，需要保证格子中频数")

    def test(self):
        cross_table = self._get_cross_table()
        condition_check_result = self._check_condition(cross_table)
        if condition_check_result:
            logging.info(f"格子频数满足要求")
            statistic, p, dof, expected = scipy.stats.chi2_contingency(cross_table)
            logging.info(f"origin: \n{cross_table}")
            logging.info(f"statistic: {statistic}")
            logging.info(f"p: {p}")
            logging.info(f"degree of freedom: {dof}")
            logging.info(f"expected: \n{expected}")
            return ChiSquareResult(True, p, p <= self.p_thread)
        else:
            logging.info(f"格子数量不满足要求，使用Fisher exact test")
            statistic, p = scipy.stats.fisher_exact(cross_table)
            return ChiSquareResult(True, p, p <= self.p_thread)

    def _get_cross_table(self):
        type_column = np.concatenate(
            [np.zeros(self.array_1.shape[0]), np.ones(self.array_2.shape[0])]
        )
        value_column = np.concatenate([self.array_1, self.array_2])
        return pd.crosstab(type_column, value_column).fillna(0)

    @classmethod
    def _check_condition(cls, cross_table: pd.DataFrame) -> bool:
        if cross_table.values.sum() <= 40:
            logging.warning(f"总频数小于40")
            return False

        if (cross_table < 5).values.sum() / (
            cross_table.shape[0] * cross_table.shape[1]
        ) >= 0.2:
            logging.warning(f"小于5的格子数超过20%")
            return False

        return True
