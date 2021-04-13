import scipy.stats
import numpy as np


class DistributionSampling:
    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    @classmethod
    def generate_normal_samplings(
        cls, mu: float, sigma: float, size: int
    ) -> np.ndarray:
        # 生成正态分布数据
        return scipy.stats.norm.rvs(loc=mu, scale=sigma, size=size)

    @classmethod
    def generate_exponential_samplings(
        cls, scale: float = 1.0, size: int = 1000
    ) -> np.ndarray:
        # 生成指数分布数据
        return np.random.exponential(scale, size)

    @classmethod
    def generate_int_samplings(cls, low: int, high: int, size: int) -> np.ndarray:
        # 生成普通离散数据
        return np.random.randint(low, high, size)

    @classmethod
    def generate_bernoulli_samplings(cls, p: float, size: int) -> np.ndarray:
        # 生成伯努利离散数据
        return scipy.stats.bernoulli.rvs(p=p, size=size)

    @classmethod
    def generate_binomial_sampling(cls, n: int, p: float, size: int) -> np.ndarray:
        # 生成二项分布数据
        return scipy.stats.binom.rvs(n=n, p=p, size=size)
