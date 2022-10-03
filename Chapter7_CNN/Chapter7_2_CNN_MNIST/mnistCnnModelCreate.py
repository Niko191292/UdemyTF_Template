import numpy as np
from typing import Tuple
from keras.datasets import mnist
from keras.layers import Activation, Dense, Conv2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.pooling import MaxPooling2D


def prepare_dataset(num_classes: int) -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = x_test.astype(np.float32)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = to_categorical(y_train, num_classes=num_classes, dtype=np.float32)
    y_test = to_categorical(y_test, num_classes=num_classes, dtype=np.float32)

    return (x_train, y_train), (x_test, y_test)


def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=3, input_shape=img_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=32, kernel_size=3))
    model.add(Flatten())
    model.add(Dense(units=num_classes))
    model.add(Activation("softmax"))
    model.summary()

    return model


if __name__ == "__main__":

    img_shape = (28, 28, 1)
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = prepare_dataset(num_classes)

    model = build_model(img_shape=img_shape, num_classes=num_classes)  # 1 Model erstellen

    model.compile(  # 2 Compiling
        loss="categorical_crossentropy",  # Für mehr als zwei Klassen
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"]
    )

    model.fit(  # 3 Training durchführen
        x=x_train,
        y=y_train,
        epochs=5,
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
