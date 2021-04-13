import unittest

from data_mocker.distribution_sampling import DistributionSampling
from hypothesis_testing.student_t_test.entity.t_test_constants import TTestAlternative
from hypothesis_testing.student_t_test.student_t_test_independent_two_variables import (
    StudentTTestIndependentTwoVariables,
)


class TestStudentTTestIndependenceTwoVariables(unittest.TestCase):
    def test_two_variables(self):
        """
        检查正态分布的假设检验结果
        """
        array_1 = DistributionSampling(seed=2).generate_normal_samplings(
            mu=3, sigma=3, size=1000
        )
        array_2 = DistributionSampling(seed=3).generate_normal_samplings(
            mu=3, sigma=4, size=1000
        )
        array_3 = DistributionSampling(seed=4).generate_normal_samplings(
            mu=4, sigma=3, size=1000
        )
        array_4 = DistributionSampling(seed=5).generate_normal_samplings(
            mu=4, sigma=3, size=1000
        )
        array_5 = DistributionSampling(seed=6).generate_normal_samplings(
            mu=5, sigma=3, size=1000
        )
        # 检查双侧结果，方差一致
        self._check_alternative(array_3, array_4, TTestAlternative.TWO_SIDED)

        # 检查双侧结果，方差不一致
        self._check_alternative(array_1, array_2, TTestAlternative.TWO_SIDED)

        # 检查LESS
        self._check_alternative(array_3, array_1, TTestAlternative.LESS)

        # 检查more
        self._check_alternative(array_4, array_5, TTestAlternative.GREATER)

    def _check_alternative(self, array_1, array_2, alternative):
        t_test_result = StudentTTestIndependentTwoVariables(
            array_1, array_2, 0.05, alternative
        ).test()
        self.assertTrue(t_test_result.condition_satisfied)
        self.assertTrue(not t_test_result.rejected)
