import unittest

from hypothesis_testing.distribution_testing.normal_distribution_test import (
    NormalDistributionTest,
)
from tests.test_hypothesis_testing.data_mocker.distribution_sampling import (
    DistributionSampling,
)


class TestDistributionTest(unittest.TestCase):
    """
    测试正态分布的假设检验效果
    """

    COMMON_SAMPLINGS_SIZE = 1000

    def test_normal_distribution_test(self):
        # 检查数据少于50正态分布是否为正态分布
        self.assertTrue(
            NormalDistributionTest.test_normal_distribution(
                DistributionSampling.generate_normal_samplings(3, 5, 30)
            )
        )
        # 检查数据50-300正态分布是否为正态分布
        self.assertTrue(
            NormalDistributionTest.test_normal_distribution(
                DistributionSampling.generate_normal_samplings(3, 5, 200)
            )
        )
        # 检查数据大于300正态分布是否为正态分布
        self.assertTrue(
            NormalDistributionTest.test_normal_distribution(
                DistributionSampling.generate_normal_samplings(3, 5, 1000)
            )
        )
        # 检查数据小于50指数分布是否不是正态分布
        self.assertTrue(
            not NormalDistributionTest.test_normal_distribution(
                DistributionSampling.generate_exponential_samplings(size=30)
            )
        )
        # 检查数据50-300指数分布是否不是正态分布
        self.assertTrue(
            not NormalDistributionTest.test_normal_distribution(
                DistributionSampling.generate_exponential_samplings(size=200)
            )
        )
        # 检查数据大于300指数分布是否不是正态分布
        self.assertTrue(
            not NormalDistributionTest.test_normal_distribution(
                DistributionSampling.generate_exponential_samplings(size=1000)
            )
        )
