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
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric

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

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanAbsoluteError()
    metric = tf.keras.metrics.BinaryAccuracy()

    p = Model(optimizer, loss, metric)
    p.train(x, y, epochs = 10)

