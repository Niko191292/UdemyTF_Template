import tensorflow as tf
import numpy as np
from typing import Tuple
from typing import List


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """XOR - Dataset"""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    return x, y


def dense(W, b, x) -> tf.Tensor:
    """Output = W*x+b"""
    """Gewichte * x + bias"""
    return tf.math.add(tf.linalg.matmul(x, W), b)


class Model:
    def __init__(
            self,
            optimizer: tf.keras.optimizers.Optimizer,
            loss: tf.keras.losses.Loss,
            metric: tf.keras.metrics.Metric,
            units_list: List[int]) -> None:
        # Model parameters
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.units_list = units_list
        # Weights
        w1_shape = [self.units_list[0], self.units_list[1]]
        self.W1 = tf.Variable(tf.random.truncated_normal(shape=w1_shape, stddev=0.1), name="W1")
        w2_shape = [self.units_list[1], self.units_list[2]]
        self.W2 = tf.Variable(tf.random.truncated_normal(shape=w2_shape, stddev=0.1, dtype=tf.float32), name="W2")
        # Biases
        b1_shape = [self.units_list[1]]
        self.b1 = tf.Variable(tf.constant(0.0, shape=b1_shape), name="b1")
        b2_shape = [self.units_list[2]]
        self.b2 = tf.Variable(tf.constant(0.0, shape=b2_shape), name="b2")
        # Trainable Variables
        self.variables = [self.W1, self.W2, self.b1, self.b2]

    def _update_weights(self, x: np.ndarray, y: np.ndarray) -> tf.Tensor:
        # Klasse um Gradienten zu berechnen
        with tf.GradientTape() as tape:
            y_pred = self.predict(x)
            loss_value = self.loss(y, y_pred)
        gradients = tape.gradient(loss_value, self.variables)
        self.optimizer.apply_gradients(zip(gradients, self.variables))
        return loss_value

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        for epoch in range(1, epochs + 1):
            # Loss Value
            loss_value = self._update_weights(x, y).numpy()
            # Metric Value Genauigkeitswert
            y_pred = self.predict(x)
            self.metric.reset_states()
            self.metric.update_state(y, y_pred)
            metric_value = self.metric.result().numpy()
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} \t Loss: {loss_value} \t Metric: {metric_value}")

    def predict(self, x: np.ndarray) -> tf.Tensor:
        """Forward Path"""
        input_layer = x # Feature Vector
        hidden_layer = dense(self.W1, self.b1, input_layer)
        hidden_layer_act = tf.nn.tanh(hidden_layer) # Activierungsfunktion tanh
        output_layer = dense(self.W2, self.b2, hidden_layer_act) # Gewichtung von hidden zum Outputlayer
        output_layer_act = tf.nn.sigmoid(output_layer) # Activierungsfunktion sigmoid
        return output_layer_act

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        y_pred = self.predict(x)
        # Loss Value
        loss_value = self.loss(y, y_pred).numpy()
        # Metric Value - Genauigkeitswert
        self.metric.reset_states()
        self.metric.update_state(y, y_pred)
        metric_value = self.metric.result().numpy()
        return [loss_value, metric_value]


if __name__ == "__main__":
    x, y = get_dataset()

    num_feature = 2
    num_targets = 1
    learning_rate = 0.5
    units_list = [num_feature, 6, num_targets] # 6 Neuronen im hidden Layer

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate) # Passt die Gewichte im Training an
    loss = tf.keras.losses.MeanAbsoluteError()
    metric = tf.keras.metrics.BinaryAccuracy()

    model = Model(optimizer, loss, metric, units_list)
    model.fit(x, y, epochs=100)
