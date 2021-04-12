import logging
import numpy as np
import scipy.stats

from hypothesis_testing.distribution_test.normal_distribution_test import (
    NormalDistributionTest,
)
from hypothesis_testing.student_t_test.entity.t_test_constants import TTestAlternative
from hypothesis_testing.student_t_test.entity.t_test_result import TTestResult


class StudentTTestIndependentTwoVariables:
    """
    双独立样本t检验。笔者没有发现好用的可以对数据进行独立性检验的方式。
    所以这个检验的独立性只能通过实验的设计来完成。
    """

    def __init__(
        self,
        array_1: np.ndarray,
        array_2: np.ndarray,
        p_thread=0.05,
        alternative: TTestAlternative = TTestAlternative.TWO_SIDED,
    ):
        """
        :param array_1: np.ndarray
        :param array_2: np.ndarray
        :param p_thread: float
        :param alternative: {‘two-sided’, ‘less’, ‘greater’}, optional[TTestAlternative]
        """
        self.array_1 = array_1
        self.array_2 = array_2
        self.p_thread = p_thread
        self.alternative = alternative
        logging.basicConfig(
            format="%(asctime)s - %(message)s", level=logging.INFO,
        )
        logging.info(f"{'StudentTTestIndependentTwoSample':=^50}")
        logging.info(f"双样本独立性检验务必要完善实验设计，保证双样本是互相独立的")

    def t_test(self) -> TTestResult:
        if not NormalDistributionTest.test_normal_distribution(self.array_1):
            logging.info(f"array_1 不符合似正态分布")
            return TTestResult(
                condition_satisfied=False, p_value=None, rejected=None
            )
        if not NormalDistributionTest.test_normal_distribution(self.array_2):
            logging.info(f"array_2 不符合似正态分布")
            return TTestResult(
                condition_satisfied=False, p_value=None, rejected=None
            )
        stat, p = scipy.stats.levene(self.array_1, self.array_2)
        logging.info(f"Levene统计量为: {stat}, p为: {p}")
        if p < 0.05:
            logging.info(f"二者方差不一致")
            return TTestResult(
                condition_satisfied=False, p_value=None, rejected=None
            )
        statistic, p = scipy.stats.ttest_ind(
            self.array_1, self.array_2, alternative=str(self.alternative.value)
        )
        # 返回是否在给定显著性水平下是否拒绝原假设
        if self.alternative == TTestAlternative.TWO_SIDED:
            logging.info(f"H0: array_1 == array_2")
            rejected = p <= self.p_thread
        elif self.alternative == TTestAlternative.LESS:
            logging.info(f"H0: array_1 < array_2")
            rejected = p <= self.p_thread
        elif self.alternative == TTestAlternative.GREATER:
            logging.info(f"H0: array_1 > array_2")
            rejected = p <= self.p_thread
        else:
            raise ValueError(f"alternative must be TTestAlternative")
        logging.info(f"reject: {rejected}")
        return TTestResult(condition_satisfied=True, p_value=p, rejected=rejected)
