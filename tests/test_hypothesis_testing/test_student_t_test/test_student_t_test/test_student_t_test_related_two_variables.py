import unittest

from data_mocker.distribution_sampling import DistributionSampling
from hypothesis_testing.student_t_test.entity.t_test_constants import TTestAlternative
from hypothesis_testing.student_t_test.student_t_test_related_two_variables import (
    StudentTTestRelatedTwoVariables,
)


class TestStudentTTestDependencePairedVariables(unittest.TestCase):
    def test_two_variables(self):
        """
        检查正态分布的假设检验结果，这里暂时直接将随机数排序当做配对样本。
        """
        array_1 = DistributionSampling.generate_normal_samplings(
            mu=3, sigma=3, size=1000
        )
        array_2 = array_1 + DistributionSampling.generate_normal_samplings(
            mu=0, sigma=1, size=1000
        )
        array_3 = array_1 + DistributionSampling.generate_normal_samplings(
            mu=0.5, sigma=3, size=1000
        )
        # 检查双侧结果
        self.assertTrue(
            StudentTTestRelatedTwoVariables(
                array_1, array_2, 0.05, TTestAlternative.TWO_SIDED
            )
            .t_test()
            .condition_satisfied
        )
        self.assertTrue(
            StudentTTestRelatedTwoVariables(
                array_1, array_3, 0.05, TTestAlternative.TWO_SIDED
            )
            .t_test()
            .condition_satisfied
        )
