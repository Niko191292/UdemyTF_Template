import os
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam

MODELS_DIR = os.path.abspath("C:/Users/nikol/Masterthesis_SOH_Parameter_Modell/UdemyTF_Template/models")
print(MODELS_DIR)
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_FILE_PATH = os.path.join(MODELS_DIR, "mnist_model.h5")
LOGS_DIR = os.path.abspath("C:/Users/nikol/Masterthesis_SOH_Parameter_Modell/UdemyTF_Template/logs")
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "mnist_model_log")


def prepare_dataset(num_features: int, num_targets: int):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, num_features).astype(np.float32)
    x_test = x_test.reshape(-1, num_features).astype(np.float32)

    y_train = to_categorical(y_train, num_classes=num_targets, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_targets, dtype=np.float32)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def build_model(num_features: int, num_targets: int) -> Sequential:
    init_w = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
    init_b = tf.keras.initializers.Constant(value=0.0)

    model = Sequential()
    # 1 Hidden Layer
    model.add(Dense(units=500, kernel_initializer=init_w,
                    bias_initializer=init_b, input_shape=(num_features,)))
    model.add(Activation("relu"))
    # 2 Hidden Layer
    model.add(Dense(units=250, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))
    # 3 Hidden Layer
    model.add(Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("relu"))

    model.add(Dense(units=num_targets, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("softmax"))
    model.summary()

    return model


if __name__ == "__main__":

    num_features = 784
    num_targets = 10

    (x_train, y_train), (x_test, y_test) = prepare_dataset(num_features, num_targets)

    model = build_model(num_features=num_features, num_targets=num_targets)  # 1 Model erstellen

    model.load_weights(MODEL_FILE_PATH)

    model.compile(  # 2 Compiling
        loss="categorical_crossentropy",  # Für mehr als zwei Klassen
        optimizer=Adam(learning_rate=0.002),
        metrics=["accuracy"]
    )

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR,
        histogram_freq=1,
        write_graph=True
    )

    model.fit(  # 3 Training durchführen
        x=x_train,
        y=y_train,
        epochs=5,
        batch_size=128,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[tb_callback]
    )

    scores = model.evaluate(  # 4 Model testen
        x=x_test,
        y=y_test,
        verbose=0
    )

    print(f"Scores before saving: {scores}")

    model.save_weights(MODEL_FILE_PATH)
