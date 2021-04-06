import scipy.stats
import numpy as np

from hypothesis_testing.distribution_testing.normal_distribution_test import (
    NormalDistributionTest,
)
from hypothesis_testing.student_t_test.entity.t_test_constants import TTestAlternative
from hypothesis_testing.student_t_test.entity.t_test_result import TTestResult


class StudentTTestOneSample:
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

    def t_test(self) -> TTestResult:
        is_normal_distribution = NormalDistributionTest.test_normal_distribution(
            self.array
        )
        if not is_normal_distribution:
            return TTestResult(condition_satisfied=False, p_value=None, rejected=None)
        statistic, p = scipy.stats.ttest_1samp(
            self.array, self.population_mean, alternative=str(self.alternative.value)
        )
        print(statistic)
        # 返回是否在给定显著性水平下是否拒绝原假设
        if self.alternative == TTestAlternative.TWO_SIDED:
            rejected = p <= self.p_thread
        elif self.alternative == TTestAlternative.LESS:
            rejected = p <= self.p_thread
        elif self.alternative == TTestAlternative.GREATER:
            rejected = p <= self.p_thread
        else:
            raise ValueError(f"alternative must be TTestAlternative")
        return TTestResult(condition_satisfied=True, p_value=p, rejected=rejected)
