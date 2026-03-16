import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from model import build_model

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

def spec_augment(spec):

    spec = spec.copy()

    num_mel = spec.shape[0]
    num_time = spec.shape[1]

    # frequency mask
    f = np.random.randint(0, 15)
    f0 = np.random.randint(0, num_mel - f)
    spec[f0:f0+f, :] = 0

    # time mask
    t = np.random.randint(0, 20)
    t0 = np.random.randint(0, num_time - t)
    spec[:, t0:t0+t] = 0

    return spec

X_PATH = "data/processed/features_cnn_bilstm.npy"
Y_PATH = "data/processed/labels_cnn_bilstm.npy"


def main():

    X = np.load(X_PATH)
    Y = np.load(Y_PATH)

    # Compute class weights
    classes = np.unique(Y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=Y
    )

    class_weights = dict(zip(range(len(classes)), weights))

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(Y.reshape(-1,1)).toarray()

    X = X[..., np.newaxis]

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.25,
        random_state=0
    )

    # Apply SpecAugment only to training samples
    for i in range(len(x_train)):
        if np.random.rand() < 0.5:
            x_train[i] = spec_augment(x_train[i])

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
        callbacks=[rlrp, early_stop],
        class_weight=class_weights
    )
    model.save("outputs/models/cnn_bilstm/model_cnn_bilstm.keras")

if __name__ == "__main__":
    main()