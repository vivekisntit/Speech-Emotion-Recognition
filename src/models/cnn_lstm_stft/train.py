import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.optimizers import SGD

from src.models.cnn_lstm_stft.model import build_cnn_model


# paths
FEATURE_PATH = "data/processed/features_stft.npy"
LABEL_PATH = "data/processed/labels_stft.npy"

MODEL_PATH = "outputs/models/cnn_lstm_stft/cnn_model.keras"


def load_dataset():

    X = np.load(FEATURE_PATH)
    y = np.load(LABEL_PATH)

    return X, y


def train():

    X, y = load_dataset()

    print("Dataset shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_cnn_model()

    optimizer = SGD(
    learning_rate=0.001,
    momentum=0.8,
    clipnorm=1.0
)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32
    )

    # evaluate
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred)

    print("\nTest Accuracy:", acc)
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    os.makedirs("outputs/models/cnn_lstm_stft", exist_ok=True)

    model.save(MODEL_PATH)

    print("\nModel saved →", MODEL_PATH)


if __name__ == "__main__":
    train()