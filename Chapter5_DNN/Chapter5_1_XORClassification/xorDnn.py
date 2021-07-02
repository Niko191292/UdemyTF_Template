import tensorflow as tf
import numpy as np
from typing import Tuple


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """XOR - Dataset"""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return x, y


class Model:
    def __init__(self, optimizer, loss, metric) -> None:
        # Model parameters
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        # Weights
        self.W1 = tf.Variable(tf.random.truncated_normal(shape=[], stddev=0.1), name="W1")
        self.W2 = tf.Variable(tf.random.truncated_normal(shape=[], stddev=0.1), name="W2")
        # Biases
        self.b1 = tf.Variable(tf.constant(0.0, shape=[]), name="b1")
        self.b2 = tf.Variable(tf.constant(0.0, shape=[]), name="b2")

    def _update_weights(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        pass

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        for epoch in range(1, epochs + 1):
            continue

    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        pass


if __name__ == "__main__":
    x, y = get_dataset()

    num_feature = 2
    num_targets = 1
    learning_rate = 0.01
    units_list = [num_feature, 6, num_targets]

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanAbsoluteError()
    metric = tf.keras.metrics.BinaryAccuracy()

    model = Model(optimizer, loss, metric)
    model.fit(x, y, epochs = 10)

