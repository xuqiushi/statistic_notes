import scipy.stats
import numpy as np

from hypothesis_testing.distribution_test.normal_distribution_test import (
    NormalDistributionTest,
)
from hypothesis_testing.student_t_test.entity.t_test_constants import TTestAlternative
from hypothesis_testing.student_t_test.entity.t_test_result import TTestResult
import logging


class StudentTTestOneVariable:
    """
    1. 方差未知
    2. 正态或近似正态
    """

    def __init__(
        self,
        array: np.ndarray,
        population_mean,
        p_thread=0.05,
        alternative: TTestAlternative = TTestAlternative.TWO_SIDED,
    ):
        """
        :param array: np.ndarray
        :param population_mean: float
        :param p_thread: float
        :param alternative: {‘two-sided’, ‘less’, ‘greater’}, optional[TTestAlternative]
        """
        self.array = array
        self.population_mean = population_mean
        self.p_thread = p_thread
        self.alternative = alternative
        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            level=logging.INFO,
        )
        logging.info(f"{'StudentTTestOneSample':=^50}")

    def test(self) -> TTestResult:
        is_normal_distribution = NormalDistributionTest.test_normal_distribution(
            self.array
        )
        if not is_normal_distribution:
            logging.info(f"当前数据不符合似正态分布")
            return TTestResult(condition_satisfied=False, p_value=None, rejected=None)
        logging.info(f"当前数据符合似正态分布")
        statistic, p = scipy.stats.ttest_1samp(
            self.array, self.population_mean, alternative=str(self.alternative.value)
        )
        logging.info(f"t test t value: {statistic}, t test p value: {p}")
        # 返回是否在给定显著性水平下是否拒绝原假设
        if self.alternative == TTestAlternative.TWO_SIDED:
            logging.info(f"H0: population expectation == {self.population_mean}")
            rejected = p <= self.p_thread
        elif self.alternative == TTestAlternative.LESS:
            logging.info(f"H0: population expectation < {self.population_mean}")
            rejected = p <= self.p_thread
        elif self.alternative == TTestAlternative.GREATER:
            logging.info(f"H0: population expectation > {self.population_mean}")
            rejected = p <= self.p_thread
        else:
            raise ValueError(f"alternative must be TTestAlternative")
        logging.info(f"reject: {rejected}")
        return TTestResult(condition_satisfied=True, p_value=p, rejected=rejected)


if __name__ == "__main__":
    StudentTTestOneVariable(np.array([1, 2, 3]), 2).test()
