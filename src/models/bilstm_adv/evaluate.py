import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import load_model

MODEL_PATH = "outputs/models/bilstm_adv/model_bilstm_adv.keras"
FEATURES_PATH = "data/processed/features_bilstm_adv.npy"
LABELS_PATH = "data/processed/labels_bilstm_adv.npy"


def load_data():
    if FEATURES_PATH.endswith(".npy"):
        X = np.load(FEATURES_PATH)
        Y = np.load(LABELS_PATH)

    else:
        df = pd.read_csv(FEATURES_PATH)
        X = df.iloc[:, :-1].values
        Y = df.iloc[:, -1].values

    encoder = OneHotEncoder()
    Y_encoded = encoder.fit_transform(Y.reshape(-1,1)).toarray()
    return X, Y, Y_encoded, encoder

def prepare_data(X, Y_encoded):
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y_encoded,
        test_size=0.25,
        random_state=0,
        shuffle=True
    )
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return x_test, y_test

def evaluate():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    
    print("Loading features...")
    X, Y, Y_encoded, encoder = load_data()

    print("Preparing data...")
    x_test, y_test = prepare_data(X, Y_encoded)

    print("Running predictions...")
    preds = model.predict(x_test)

    y_pred = encoder.inverse_transform(preds)
    y_true = encoder.inverse_transform(y_test)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("\n MODEL PERFORMANCE ")
    print("Accuracy :", acc)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\n CLASSIFICATION REPORT ")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    evaluate()