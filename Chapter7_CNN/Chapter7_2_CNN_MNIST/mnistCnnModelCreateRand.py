import os
import shutil
import numpy as np
import tensorflow as tf
from typing import Tuple
from tensorflow.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input
from scipy.stats import randint, uniform
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import ParameterSampler
from tf_utils.cifarDataAdvanced import CIFAR10


np.random.seed(0)
tf.random.set_seed(0)

LOGS_DIR = os.path.abspath("C:/Users/nikol/Masterthesis_SOH_Parameter_Modell/UdemyTF_Template/logs")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


def build_model(
    img_shape: Tuple[int, int, int],
    num_classes: int,
    optimizer: tf.keras.optimizers.Optimizer,
    learning_rate: float,
    filter_block1: int,
    kernel_size_block1: int,
    filter_block2: int,
    kernel_size_block2: int,
    filter_block3: int,
    kernel_size_block3: int,
    dense_layer_size: int
) -> Model:

    input_img = Input(shape=img_shape)

    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding="same")(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block1, kernel_size=kernel_size_block1, padding="same")(input_img)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding="same")(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block2, kernel_size=kernel_size_block2, padding="same")(input_img)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding="same")(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=filter_block3, kernel_size=kernel_size_block3, padding="same")(input_img)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(units=dense_layer_size)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(
        inputs=[input_img],
        outputs=[y_pred]
    )

    opt = optimizer(learning_rate=learning_rate)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    data = CIFAR10()

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()
    test_dataset = data.get_test_set()

    epochs = 15
    batch_size = 1042

    param_distribution = {
        "optimizer": [Adam, RMSprop],
        "learning_rate": uniform(0.001, 0.0001),
        "filter_block1": randint(16, 64),
        "kernel_size_block1": randint(3, 7),
        "filter_block2": randint(16, 64),
        "kernel_size_block2": randint(3, 7),
        "filter_block3": randint(16, 64),
        "kernel_size_block3": randint(3, 7),
        "dense_layer_size": randint(128, 1024),
    }

    results = {
        "best_score": -np.inf,
        "best_params": {},
        "val_scores": [],
        "params": []
    }

    n_models = 4
    dist = ParameterSampler(param_distribution, n_iter=n_models)

    print(f"Param Combination: {len(dist)}")

    for idx, comb in enumerate(dist):
        print(f"RunningComb: {idx}")

        model = build_model(
            data.img_shape,
            data.num_classes,
            **comb
        )

        model_log_dir = os.path.join((LOGS_DIR + "ModelRands"), f"modelRand{idx}")
        if os.path.exists(model_log_dir):
            shutil.rmtree(model_log_dir)
            os.mkdir(model_log_dir)

        tb_callback = TensorBoard(
            log_dir=model_log_dir,
            histogram_freq=0
        )

        model.fit(
            train_dataset,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=val_dataset,
            callbacks=[tb_callback]
        )

        val_accuracy = model.evaluate(
            val_dataset,
            batch_size=batch_size,
            verbose=0,

        )[1]
        results["val_scores"].append(val_accuracy)
        results["params"].append(comb)

    best_run_idx = np.argmax(results["val_scores"])
    results["best_score"] = results["val_scores"][best_run_idx]
    results["best_params"] = results["params"][best_run_idx]

    print(f"Best score: {results['best_score']} using params: {results['best_params']}\n")

    scores = results["val_ scores"]
    params = results["params"]

    for score, param in zip(scores, params):
        print(f"Score: {score} with param {param}")
