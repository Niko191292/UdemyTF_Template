import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.initializers import Constant
from keras.datasets import boston_housing
from keras.initializers import RandomUniform
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
# from tensorflow.python.ops.math_ops import reduce_sum


def r_squared(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''_summary_

    Args:
        y_true (tf.Tensor): _description_
        y_pred (tf.Tensor): _description_

    Returns:
        tf.Tensor: _description_
    '''
    error = tf.math.subtract(y_true, y_pred)
    squared_err = tf.math.square(error)
    numerator = tf.math.reduce_sum(squared_err)
    y_true_mean = tf.math.reduce_mean(y_true)
    mean_dev = tf.math.subtract(y_true, y_true_mean)
    squared_mean_dev = tf.math.square(mean_dev)
    denominator = tf.reduce_sum(squared_mean_dev)
    R2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_clipped = tf.clip_by_value(R2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped


def build_model(num_features: int, num_targets: int) -> Sequential:
    init_w = RandomUniform(minval=-1.0, maxval=1.0)
    init_b = Constant(value=0.0)

    model = Sequential()
    model.add(Dense(units=16, kernel_initializer=init_w,
                    bias_initializer=init_b, input_shape=(num_features,)))
    model.add(Activation("relu"))
    model.add(Dense(units=num_targets, kernel_initializer=init_w, bias_initializer=init_b))
    model.summary()

    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    x_train = x_train.astype(np.float32)  # Immer 32 bit floats
    y_train = y_train.astype(np.float32)  # Immer 32 bit floats
    x_test = x_test.astype(np.float32)  # Immer 32 bit floats
    y_test = y_test.astype(np.float32)  # Immer 32 bit floats

    y_train = np.reshape(y_train, newshape=(-1, 1))
    y_test = np.reshape(y_test, newshape=(-1, 1))

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    num_features = x_train.shape[1]
    num_targets = y_train.shape[1]

    model = build_model(num_features=num_features, num_targets=num_targets)  # 1 Model erstellen

    adam = tf.keras.optimizers.Adam(
        learning_rate=0.05
    )

    model.compile(  # 2 Compiling
        loss="mse",  # Mean Squared Error
        optimizer=adam,
        metrics=[r_squared]
    )

    model.fit(  # 3 Training durchführen
        x=x_train,
        y=y_train,
        epochs=5_00,
        batch_size=128,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    scores = model.evaluate(  # 4 Model testen
        x=x_test,
        y=y_test,
        verbose=0
    )

    print(scores)
