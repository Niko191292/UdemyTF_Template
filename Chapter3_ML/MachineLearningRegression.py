import matplotlib.pyplot as plt
import numpy as np

from tf_utils.dummyData import regression_data


def model(x: np.ndarray) -> np.ndarray:
    m = 1.0  # slope
    b = 0.0  # intercept

    return m * x + b


def main() -> None:
    x, y = regression_data()

    y_pred = model(x)

    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.show()


if __name__ == "__main__":
    main()
