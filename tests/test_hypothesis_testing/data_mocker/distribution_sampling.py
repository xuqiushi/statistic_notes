import numpy as np


class DistributionSampling:
    @classmethod
    def generate_normal_samplings(
        cls, mu: float, sigma: float, size: int
    ) -> np.ndarray:
        # 生成正态分布数据
        return np.random.normal(mu, sigma, size)

    @classmethod
    def generate_exponential_samplings(
        cls, scale: float = 1.0, size: int = 1000
    ) -> np.ndarray:
        # 生成指数分布数据
        return np.random.exponential(scale, size)
