import unittest
import numpy as np

from hypothesis_testing.student_t_test.normal_distribution_test import (
    NormalDistributionTestCollection,
)


def generate_normal_samplings(mu: float, sigma: float, size: float) -> np.ndarray:
    # 生成正态分布数据
    return np.random.normal(mu, sigma, size)


def generate_exponential_samplings(
    scale: float = 1.0, size: float = 1000
) -> np.ndarray:
    # 生成指数分布数据
    return np.random.exponential(scale, size)


class TestDivision(unittest.TestCase):
    COMMON_SAMPLINGS_SIZE = 1000

    def test_test_normal_distribution(self):
        self.assertTrue(
            NormalDistributionTestCollection.test_normal_distribution(
                generate_normal_samplings(0, 1, 30)
            )
        )
        self.assertTrue(
            NormalDistributionTestCollection.test_normal_distribution(
                generate_normal_samplings(0, 1, 1000)
            )
        )
        self.assertTrue(
            not NormalDistributionTestCollection.test_normal_distribution(
                generate_exponential_samplings(size=30)
            )
        )
        self.assertTrue(
            not NormalDistributionTestCollection.test_normal_distribution(
                generate_exponential_samplings(size=1000)
            )
        )
