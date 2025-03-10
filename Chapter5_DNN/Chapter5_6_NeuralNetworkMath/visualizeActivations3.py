from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


def f(x: float) -> float:
    return x**4 + -5 * x**3 + 14 * x**2 + x + 10


def relu(x: float) -> float:
    if x > 0:
        return x
    else:
        return 0


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(start=-10.0, stop=10.0, num=1000).reshape(-1, 1)
    y = f(x)  # type: ignore
    return x, y  # type: ignore


def build_model1() -> Sequential:
    model = Sequential()
    model.add(Dense(units=200))  # Input zu Hidden
    model.add(Dense(units=200))  # Input zu Hidden
    model.add(Dense(units=1))  # Vom Hidden zum Output
    return model


def build_model2() -> Sequential:
    model = Sequential()
    model.add(Dense(units=500))  # Input zu Hidden
    model.add(Activation("relu"))  # ReLU vom Hidden
    model.add(Dense(units=500))  # Input zu Hidden
    model.add(Activation("relu"))  # ReLU vom Hidden
    model.add(Dense(units=1))  # Vom Hidden zum Output
    return model


def build_model3() -> Sequential:
    model = Sequential()
    model.add(Dense(units=500))  # Input zu Hidden
    model.add(Activation("sigmoid"))  # ReLU vom Hidden
    model.add(Dense(units=500))  # Input zu Hidden
    model.add(Activation("sigmoid"))  # ReLU vom Hidden
    model.add(Dense(units=1))  # Vom Hidden zum Output
    return model


def main() -> None:
    x, y = get_dataset()

    model1 = build_model1()
    model1.compile(optimizer=Adam(learning_rate=1e-2), loss="mse")
    model1.fit(x, y, epochs=30)
    y_pred_linear = model1.predict(x)

    model2 = build_model2()
    model2.compile(optimizer=Adam(learning_rate=1e-2), loss="mse")
    model2.fit(x, y, epochs=30)
    y_pred_relu = model2.predict(x)

    model3 = build_model3()
    model3.compile(optimizer=Adam(learning_rate=1e-2), loss="mse")
    model3.fit(x, y, epochs=30)
    y_pred_sigmoid = model3.predict(x)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(24, 12))
    plt.grid(True)
    ax1.plot(x, y, color="blue")
    ax1.plot(x.flatten(), y_pred_linear.flatten(), color="red")
    ax2.plot(x, y, color="blue")
    ax2.plot(x.flatten(), y_pred_sigmoid.flatten(), color="red")
    ax3.plot(x, y, color="blue")
    ax3.plot(x.flatten(), y_pred_relu.flatten(), color="red")
    plt.show()


if __name__ == "__main__":
    main()
