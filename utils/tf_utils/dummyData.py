from typing import Tuple
import numpy as np

np.random.seed(0)


def f(x: np.ndarray) -> np.ndarray:
    return 2.0 * x + 5.0


def classification_data() -> Tuple[np.ndarray, np.ndarray]:
    N = 30
    n_class1 = N // 2
    n_class2 = N // 2
    x1mean = np.array([5.0, 0.0])
    x1cov = np.array([[3.0, 0.0], [0.0, 1.0]])
    x1 = np.random.multivariate_normal(mean=x1mean, cov=x1cov, size=n_class1)
    y1 = np.array([0 for _ in range(n_class1)])
    x2mean = np.array([0.0, 0.0])
    x2cov = np.array([[1.0, 0.0], [0.0, 3.0]])
    x2 = np.random.multivariate_normal(mean=x2mean, cov=x2cov, size=n_class2)
    y2 = np.array([1 for _ in range(n_class1)])
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    return x, y


def regression_data() -> Tuple[np.ndarray, np.ndarray]:
    N = 100
    x = np.random.uniform(low=-10.0, high=10.0, size=N)
    y = f(x) + np.random.normal(scale=2.0, size=100)
    return x, y


if __name__ == "__main__":
    x, y = classification_data()
