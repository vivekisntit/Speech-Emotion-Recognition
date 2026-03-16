import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ReduceLROnPlateau

import seaborn as sns
import matplotlib.pyplot as plt


FEATURES = "data/processed/features_cnn_features.csv"


def load_data():

    df = pd.read_csv(FEATURES)

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(Y.reshape(-1,1)).toarray()

    return X, Y, encoder


def prepare_data(X, Y):

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.25,
        random_state=0,
        shuffle=True
    )

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    return x_train, x_test, y_train, y_test


def build_model(input_shape, num_classes):

    model = Sequential()

    model.add(Conv1D(256, 5, padding="same", activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding="same"))

    model.add(Conv1D(256, 5, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding="same"))

    model.add(Conv1D(128, 5, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding="same"))

    model.add(Dropout(0.2))

    model.add(Conv1D(64, 5, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding="same"))

    model.add(Flatten())

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    print("Loading features...")
    X, Y, encoder = load_data()

    print("Preparing data...")
    x_train, x_test, y_train, y_test = prepare_data(X, Y)

    print("Building model...")
    model = build_model(
        input_shape=(x_train.shape[1],1),
        num_classes=y_train.shape[1]
    )

    model.summary()

    rlrp = ReduceLROnPlateau(
        monitor="loss",
        factor=0.4,
        patience=2,
        min_lr=1e-7
    )

    print("Training model...")

    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[rlrp]
    )

    print("Evaluating model...")

    score = model.evaluate(x_test, y_test)
    print("Test Accuracy:", score[1] * 100)

    os.makedirs("outputs/models/cnn_features", exist_ok=True)

    model.save("outputs/models/cnn_features/model_cnn_features.keras")

    print("Model saved to outputs/model_models/cnn_features/")


if __name__ == "__main__":
    main()