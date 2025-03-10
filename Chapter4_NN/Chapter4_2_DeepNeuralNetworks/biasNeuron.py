import os

import matplotlib.pyplot as plt
import numpy as np


def plot_function(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    offset: float = 0.0,
    name: str = "",
) -> None:
    plt.step(input_signal + offset, output_signal, color="blue")
    plt.step(input_signal, output_signal, color="black")
    plt.step(input_signal - offset, output_signal, color="red")
    plt.xlabel("a")
    plt.ylabel("f(a)")
    plt.legend([f"Bias: -{offset}", "Bias: 0", f"Bias: {offset}"])
    plt.title(f"Activation function: {name}")
    plt.show()


def main() -> None:
    input_signal = np.linspace(start=-10, stop=10, num=1000)
    offset = 2

    # Step function
    # f(a) = 0, if a <= 0 else 1
    output_signal = np.array([0 if a <= 0 else 1 for a in input_signal])
    plot_function(input_signal, output_signal, offset, name="step")

    # Tanh
    # f(a) = tanh(a) = 2 / (1+e^(-2a)) - 1
    output_signal = np.array(
        [2 / (1 + np.exp(-2 * a)) - 1 for a in input_signal]
    )
    plot_function(input_signal, output_signal, offset, name="tanh")

    # SIGMOID
    # sigmoid(a) = 1 / (1 + e^-a)
    output_signal = np.array([1 / (1 + np.exp(-a)) for a in input_signal])
    plot_function(input_signal, output_signal, offset, name="sigmoid")

    # RELU = Rectified Linear Unit
    # f(a) = max (0, a)
    output_signal = np.array([max(0, a) for a in input_signal])
    plot_function(input_signal, output_signal, offset, name="relu")


if __name__ == "__main__":
    main()
