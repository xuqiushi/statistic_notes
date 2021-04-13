import logging
import numpy as np
import scipy.stats

from hypothesis_testing.student_t_test.entity.t_test_constants import TTestAlternative
from hypothesis_testing.student_t_test.entity.t_test_result import TTestResult
from hypothesis_testing.student_t_test.student_t_test_two_variables.student_t_test_two_variables import (
    StudentTTestTwoVariables,
)


class StudentTTestIndependentTwoVariables(StudentTTestTwoVariables):
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
        super().__init__(array_1, array_2, p_thread, alternative)
        logging.info(f"{'StudentTTestIndependentTwoVariables':=^50}")
        logging.info(f"双样本独立性检验务必要完善实验设计，保证双样本是互相独立的")

    def t_test(self) -> TTestResult:
        if not self._check_normal():
            statistic, p = scipy.stats.mannwhitneyu(
                self.array_1, self.array_2, alternative=str(self.alternative.value)
            )
            return self._get_p_result(p)
        else:
            levene_p = self._check_variances_homogeneity()
            if levene_p < 0.05:
                logging.info(f"二者方差不一致")
                statistics, p = scipy.stats.ttest_ind(
                    self.array_1,
                    self.array_2,
                    equal_var=False,
                    alternative=str(self.alternative.value),
                )
            else:
                logging.info(f"二者方差一致")
                statistic, p = scipy.stats.ttest_ind(
                    self.array_1, self.array_2, alternative=str(self.alternative.value)
                )
            # 返回是否在给定显著性水平下是否拒绝原假设
            return self._get_p_result(p)
