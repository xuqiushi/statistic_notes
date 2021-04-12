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
        array_1 = DistributionSampling.generate_normal_samplings(
            mu=3, sigma=3, size=1000
        )
        array_2 = DistributionSampling.generate_normal_samplings(
            mu=3, sigma=4, size=1000
        )
        array_3 = DistributionSampling.generate_normal_samplings(
            mu=4, sigma=3, size=1000
        )
        array_4 = DistributionSampling.generate_normal_samplings(
            mu=4, sigma=3, size=1000
        )
        array_5 = DistributionSampling.generate_normal_samplings(
            mu=5, sigma=3, size=1000
        )
        # 检查双侧结果，方差一致
        self.assertTrue(
            StudentTTestIndependentTwoVariables(
                array_3, array_4, 0.05, TTestAlternative.TWO_SIDED
            )
            .t_test()
            .condition_satisfied
        )
        # 检查双侧结果，方差不一致
        self.assertTrue(
            StudentTTestIndependentTwoVariables(
                array_1, array_2, 0.05, TTestAlternative.TWO_SIDED
            )
            .t_test()
            .condition_satisfied
        )
        # 检查LESS
        self.assertTrue(
            StudentTTestIndependentTwoVariables(
                array_1, array_3, 0.05, TTestAlternative.TWO_SIDED
            )
            .t_test()
            .condition_satisfied
        )
        # 检查more
        self.assertTrue(
            StudentTTestIndependentTwoVariables(
                array_5, array_4, 0.05, TTestAlternative.TWO_SIDED
            )
            .t_test()
            .condition_satisfied
        )
