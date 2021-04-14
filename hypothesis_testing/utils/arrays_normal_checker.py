import logging

import numpy as np

from hypothesis_testing.distribution_test.normal_distribution_test import (
    NormalDistributionTest,
)


class ArraysNormalChecker:
    @classmethod
    def check_double_normal(cls, array_1: np.ndarray, array_2: np.ndarray) -> bool:
        check_result = True
        if not NormalDistributionTest.test_normal_distribution(array_1):
            logging.info(f"array_1 不符合似正态分布")
            check_result = False
        if not NormalDistributionTest.test_normal_distribution(array_2):
            logging.info(f"array_2 不符合似正态分布")
            check_result = False
        return check_result
