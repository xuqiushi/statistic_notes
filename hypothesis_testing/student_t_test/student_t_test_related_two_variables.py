import logging

import numpy as np
import scipy.stats

from hypothesis_testing.distribution_test.normal_distribution_test import (
    NormalDistributionTest,
)
from hypothesis_testing.entity.compare_variable_result import CompareVariableResult
from hypothesis_testing.student_t_test.entity.t_test_constants import TTestAlternative
from hypothesis_testing.student_t_test.student_t_test_two_variables.student_t_test_two_variables import (
    StudentTTestTwoVariables,
)


class StudentTTestRelatedTwoVariables(StudentTTestTwoVariables):
    def __init__(
        self,
        array_1: np.ndarray,
        array_2: np.ndarray,
        p_thread=0.05,
        alternative: TTestAlternative = TTestAlternative.TWO_SIDED,
    ):
        super().__init__(array_1, array_2, p_thread, alternative)
        logging.info(f"{'StudentTTestRelatedTwoVariables':=^50}")
        logging.info(f"配对独立性检验要满足量样本正态或者样本之差正态，因为默认了同一样本，所以不需要方差齐性")

    def test(self) -> CompareVariableResult:
        if not NormalDistributionTest.test_normal_distribution(
            self.array_1 - self.array_2
        ):
            if not self._check_normal():
                return CompareVariableResult(
                    condition_satisfied=False, p_value=None, rejected=None
                )
        statistic, p = scipy.stats.ttest_rel(
            self.array_1, self.array_2, alternative=str(self.alternative.value)
        )
        # 返回是否在给定显著性水平下是否拒绝原假设
        return self._get_p_result(p)
