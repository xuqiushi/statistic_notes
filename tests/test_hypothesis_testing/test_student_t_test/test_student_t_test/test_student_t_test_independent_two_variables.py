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
        array_1 = DistributionSampling(fixed_seed=True).generate_normal_samplings(
            mu=3, sigma=3, size=1000
        )
        array_2 = DistributionSampling(fixed_seed=True).generate_normal_samplings(
            mu=3, sigma=4, size=1000
        )
        array_3 = DistributionSampling(fixed_seed=True).generate_normal_samplings(
            mu=4, sigma=3, size=1000
        )
        array_4 = DistributionSampling(fixed_seed=True).generate_normal_samplings(
            mu=4, sigma=3, size=1000
        )
        array_5 = DistributionSampling(fixed_seed=True).generate_normal_samplings(
            mu=5, sigma=3, size=1000
        )
        # 检查双侧结果，方差一致
        student_t_test_two_variables = StudentTTestIndependentTwoVariables(
            array_3, array_4, 0.05, TTestAlternative.TWO_SIDED
        )
        result = student_t_test_two_variables.t_test()
        self.assertTrue(result.condition_satisfied)
        # 检查双侧结果，方差不一致
        student_t_test_two_variables = StudentTTestIndependentTwoVariables(
            array_1, array_2, 0.05, TTestAlternative.TWO_SIDED
        )
        result = student_t_test_two_variables.t_test()
        self.assertTrue(not result.condition_satisfied)
        # 检查LESS
        student_t_test_two_variables = StudentTTestIndependentTwoVariables(
            array_1, array_3, 0.05, TTestAlternative.TWO_SIDED
        )
        result = student_t_test_two_variables.t_test()
        self.assertTrue(result.condition_satisfied)
        # 检查more
        student_t_test_two_variables = StudentTTestIndependentTwoVariables(
            array_5, array_4, 0.05, TTestAlternative.TWO_SIDED
        )
        result = student_t_test_two_variables.t_test()
        self.assertTrue(result.condition_satisfied)
