from typing import Tuple
import os
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Activation, Dense, Input, SimpleRNN, LSTM, GRU, Embedding
from keras.models import Model
from keras.optimizers import Adam
from tf_utils.imdbDataAdvanced import IMDB

np.random.seed(0)
tf.random.set_seed(0)

MODEL_DIR = os.path.abspath("C:/Users/nikol/Masterthesis_SOH_Parameter_Modell/UdemyTF_Template/models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "imbd_cassification.h5")
LOGS_DIR = os.path.abspath("C:/Users/nikol/Masterthesis_SOH_Parameter_Modell/UdemyTF_Template/logs")
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
MODEL_LOG_DIR = os.path.join(LOGS_DIR, "imdb_model")


def create_rnn_model(
    input_shape: Tuple[int],
    num_classes: int,
    vocab_size: int,
    embedding_dim: int,
    sequence_length: int
) -> Model:

    input_text = Input(shape=input_shape)
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length
    )(input_text)
    x = SimpleRNN(units=80, return_sequences=False)(x)
    x = Dense(units=80)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    out = Activation("Softmax")(x)
    model = Model(inputs=[input_text], outputs=[out])
    opt = Adam(learning_rate=1e-4)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    model.summary()
    return model


def create_lstm_model(
    input_shape: Tuple[int],
    num_classes: int,
    vocab_size: int,
    embedding_dim: int,
    sequence_length: int
) -> Model:

    input_text = Input(shape=input_shape)
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length
    )(input_text)
    x = LSTM(units=80, return_sequences=False)(x)
    x = Dense(units=80)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    out = Activation("Softmax")(x)
    model = Model(inputs=[input_text], outputs=[out])
    opt = Adam(learning_rate=1e-4)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    model.summary()
    return model


def create_gru_model(
    input_shape: Tuple[int],
    num_classes: int,
    vocab_size: int,
    embedding_dim: int,
    sequence_length: int
) -> Model:

    input_text = Input(shape=input_shape)
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length
    )(input_text)
    x = GRU(units=80, return_sequences=False)(x)
    x = Dense(units=80)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    out = Activation("Softmax")(x)
    model = Model(inputs=[input_text], outputs=[out])
    opt = Adam(learning_rate=1e-4)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    model.summary()
    return model


def main() -> None:
    vocab_size = 20_000
    sequence_length = 80
    embedding_dim = 50

    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR, histogram_freq=1, write_graph=True
    )

    imdb_data = IMDB(vocab_size=vocab_size, sequence_length=sequence_length)
    train_dataset = imdb_data.get_train_set()
    val_dataset = imdb_data.get_val_set()
    test_dataset = imdb_data.get_test_set()
    input_shape = (sequence_length,)
    num_classes = imdb_data.num_classes

    batch_size = 512
    epochs = 20

    model_fns = {
        "RNN": create_rnn_model,
        "LSTM": create_lstm_model,
        "GRU": create_gru_model
    }

    for name, model_fn in model_fns.items():
        print(f"Model: {name}")
        model = model_fn(input_shape, num_classes, vocab_size, embedding_dim, sequence_length)
        model.fit(
            x=train_dataset,
            verbose=1,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[tb_callback])
        score = model.evaluate(x=test_dataset, verbose=0, batch_size=batch_size)
        print(f"Performance Test : {score}")


if __name__ == "__main__":
    main()
