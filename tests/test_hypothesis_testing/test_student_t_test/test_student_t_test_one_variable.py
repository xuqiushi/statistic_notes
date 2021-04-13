import unittest

from hypothesis_testing.student_t_test.entity.t_test_constants import TTestAlternative
from hypothesis_testing.student_t_test.student_t_test_one_variable import (
    StudentTTestOneVariable,
)
from data_mocker.distribution_sampling import DistributionSampling


class TestStudentTTestOneVariable(unittest.TestCase):
    """
    测试单样本t检验效果
    """

    def test_not_normal(self):
        """
        检验对于数据是否满足正态分布的判断是否有效
        """
        array = DistributionSampling.generate_exponential_samplings(size=1000)
        student_t_test_one_sample = StudentTTestOneVariable(
            array, 0, 0.05, TTestAlternative.TWO_SIDED
        )
        result = student_t_test_one_sample.test()
        self.assertTrue(not result.condition_satisfied)

    def test_normal_result(self):
        """
        检查正态分布的假设检验结果
        """
        array = DistributionSampling.generate_normal_samplings(mu=5, sigma=3, size=2000)
        # 检查双侧结果
        student_t_test_one_sample = StudentTTestOneVariable(
            array, 6, 0.05, TTestAlternative.TWO_SIDED
        )
        two_side_result = student_t_test_one_sample.test()
        self.assertTrue(two_side_result.condition_satisfied)
        self.assertTrue(two_side_result.rejected)
        # 检查小于的结果
        student_t_test_one_sample = StudentTTestOneVariable(
            array, 4, 0.05, TTestAlternative.LESS
        )
        less_result = student_t_test_one_sample.test()
        self.assertTrue(less_result.condition_satisfied)
        self.assertTrue(not less_result.rejected)
        # 检查大于的结果
        student_t_test_one_sample = StudentTTestOneVariable(
            array, 6, 0.05, TTestAlternative.GREATER
        )
        greater_result = student_t_test_one_sample.test()
        self.assertTrue(greater_result.condition_satisfied)
        self.assertTrue(not greater_result.rejected)


if __name__ == "__main__":
    unittest.main()
