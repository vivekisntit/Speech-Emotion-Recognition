import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    LSTM,
    TimeDistributed,
    Reshape
)

IMG_SIZE = 128
NUM_CLASSES = 7


def build_cnn_model():

    model = Sequential([

        Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),

        Flatten(),

        Dense(1024, activation="relu"),
        Dropout(0.25),

        Dense(NUM_CLASSES, activation="softmax")
    ])

    return model


def build_lstm_model():

    model = Sequential([

        Reshape((IMG_SIZE, IMG_SIZE), input_shape=(IMG_SIZE, IMG_SIZE, 1)),

        LSTM(256, return_sequences=True),
        LSTM(256, return_sequences=True),
        LSTM(256),

        Dense(256, activation="relu"),
        Dropout(0.25),

        Dense(NUM_CLASSES, activation="softmax")
    ])

    return model


def build_cnn_lstm_model():

    model = Sequential([

        TimeDistributed(
            Conv2D(32, (3,3), activation="relu"),
            input_shape=(IMG_SIZE, IMG_SIZE, 1, 1)
        ),

        TimeDistributed(MaxPooling2D((2,2))),

        TimeDistributed(
            Conv2D(64, (3,3), activation="relu")
        ),

        TimeDistributed(MaxPooling2D((2,2))),

        TimeDistributed(Flatten()),

        LSTM(512, return_sequences=True),
        LSTM(512),

        Dense(256, activation="relu"),
        Dropout(0.25),

        Dense(NUM_CLASSES, activation="softmax")
    ])

    return model