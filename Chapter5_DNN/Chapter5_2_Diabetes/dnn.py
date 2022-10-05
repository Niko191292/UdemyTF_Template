from typing import Tuple

import numpy as np
import tensorflow as tf
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def r_squared(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    error = tf.math.subtract(y_true, y_pred)
    squared_err = tf.math.square(error)
    numerator = tf.math.reduce_sum(squared_err)
    y_true_mean = tf.math.reduce_mean(y_true)
    mean_dev = tf.math.subtract(y_true, y_true_mean)
    squared_mean_dev = tf.math.square(mean_dev)
    denominator = tf.reduce_sum(squared_mean_dev)
    R2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_cliped = tf.clip_by_value(R2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_cliped


def get_dataset() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    dataset = load_diabetes()
    x = dataset.data
    y = dataset.target.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    return (x_train, y_train), (x_test, y_test)


def build_model() -> Sequential:
    '''
    Funktion ist zum modellieren des Modells zuständig.

    Returns:
        Sequential: Squenzielles Modell wird zurückgegeben.
    '''
    model = Sequential()

    return model


def main() -> None:
    '''
    Hauptfunktion, die bei beginn gestartet wird.
    '''
    (x_train, y_train), (x_test, y_test) = get_dataset()

    print(f"x_train: {x_train}")
    print(f"y_train: {y_train}")
    print(f"x_test: {x_test}")
    print(f"y_test: {y_test}")

    model = build_model()
    opt = Adam(learning_rate=0.01)
    model.compile(loss="mse", optimizer=opt, metrics=[r_squared])
    model.fit(
        x=x_train,
        y=y_train,
        epochs=3_000,
        batch_size=128,
        verbose=1,
        validation_data=(x_test, y_test))


if __name__ == "__main__":

    main()
