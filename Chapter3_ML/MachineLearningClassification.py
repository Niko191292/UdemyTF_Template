import matplotlib.pyplot as plt
import numpy as np

from tf_utils.dummyData import classification_data


def model(x: np.ndarray) -> np.ndarray:
    m = 1.0  # slope
    b = 0.0  # intercept

    return m * x + b


def main() -> None:
    x, y = classification_data()

    y_pred = model(x)

    colors = np.array(["red", "blue"])
    plt.scatter(x[:, 0], x[:, 1], color=colors[y])
    plt.plot(x, y_pred)
    plt.show()


if __name__ == "__main__":
    main()
