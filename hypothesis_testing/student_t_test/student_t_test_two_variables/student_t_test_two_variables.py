import logging
import numpy as np
import scipy.stats

from hypothesis_testing.distribution_test.normal_distribution_test import NormalDistributionTest
from hypothesis_testing.student_t_test.entity.t_test_constants import TTestAlternative
from hypothesis_testing.student_t_test.entity.t_test_result import TTestResult


class StudentTTestTwoVariables:
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

    def _get_p_result(self, p: float) -> TTestResult:
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

    def _check_normal(self) -> bool:
        check_result = True
        if not NormalDistributionTest.test_normal_distribution(self.array_1):
            logging.info(f"array_1 不符合似正态分布")
            check_result = False
        if not NormalDistributionTest.test_normal_distribution(self.array_2):
            logging.info(f"array_2 不符合似正态分布")
            check_result = False
        return check_result

    def _check_variances_homogeneity(self) -> float:
        stat, p = scipy.stats.levene(self.array_1, self.array_2)
        logging.info(f"Levene统计量为: {stat}, p为: {p}")
        return p
