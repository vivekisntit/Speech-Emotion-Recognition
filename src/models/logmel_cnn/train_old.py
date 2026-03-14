import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from model import build_model


X_PATH = "data/processed/features_logmel.npy"
Y_PATH = "data/processed/labels_logmel.npy"

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

def main():

    X = np.load(X_PATH)
    Y = np.load(Y_PATH)

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(Y.reshape(-1,1)).toarray()

    X = X[..., np.newaxis]

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.25,
        random_state=0
    )

    model = build_model(
        input_shape=X.shape[1:],
        num_classes=y_train.shape[1]
    )

    rlrp = ReduceLROnPlateau(
        monitor="loss",
        factor=0.4,
        patience=2,
        min_lr=1e-7
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=40,
        validation_data=(x_test, y_test),
        callbacks=[rlrp, early_stop]
    )

    model.save("outputs/models/logmel_cnn/logmel_cnn_v1.keras")


if __name__ == "__main__":
    main()