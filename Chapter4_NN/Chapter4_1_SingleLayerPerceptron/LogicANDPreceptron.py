from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import os

# open Tensorboard
# tensorboard --logdir logs/AND_SLP

# Step function as activation function


def step(x: tf.Tensor) -> tf.Tensor:
    return tf.where(tf.greater(x, 0), tf.ones_like(x), tf.zeros_like(x))


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """AND-Function dataset"""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)  # Input in das UND-Gatter
    y = np.array([[0], [0], [0], [1]], dtype=np.float32)  # Ergebnis eines UND-Gatters
    tf.keras.initializers.Constant(
    value=0
)
    return x, y


def build_model(num_features: int, num_targets: int) -> Sequential:
    init_w = RandomUniform(minval=-1.0, maxval=1.0)
    init_b = Constant(value=0.0)

    model = Sequential()
    model.add(Dense(units=16, kernel_initializer=init_w,
                    bias_initializer=init_b, input_shape=(num_features,)))
    model.add(Activation(step))
    model.add(Dense(units=num_targets, kernel_initializer=init_w, bias_initializer=init_b))
    model.summary()

    return model


if __name__ == "__main__":
    x, y = get_dataset()

    num_feature = 2
    num_targets = 1
    learning_rate = 0.01

    model = build_model(num_features=num_feature, num_targets=num_targets)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate)  # Passt die Gewichte im Training an
    loss = tf.keras.losses.MeanAbsoluteError()
    metric = tf.keras.metrics.BinaryAccuracy()

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    LOGS_DIR = os.path.abspath("C:/Users/nikol/UdemyTF_Template/logs")
    MODEL_LOG_DIR = os.path.join(LOGS_DIR, "AND_SLP")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=MODEL_LOG_DIR,
        histogram_freq=1,
        write_graph=True
    )

    model.fit(
        x=x,
        y=y,
        epochs=200,
        verbose=1,
        validation_data=(x, y),
        callbacks=[tensorboard_callback]
    )

    scores = model.evaluate(  # 4 Model testen
        x=x,
        y=y,
        verbose=0
    )
    print(scores)
