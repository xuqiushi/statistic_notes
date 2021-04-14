import logging
import numpy as np
import scipy.stats


class ArraysVariancesHomogeneityChecker:
    @classmethod
    def check_double_variances_homogeneity(
        cls, array_1: np.ndarray, array_2: np.ndarray
    ) -> bool:
        stat, p = scipy.stats.levene(array_1, array_2)
        logging.info(f"Levene统计量为: {stat}, p为: {p}")
        return p > 0.05
