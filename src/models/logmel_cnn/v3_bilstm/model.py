from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization


from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D,
    BatchNormalization, Dropout,
    Reshape, Bidirectional, LSTM,
    Dense
)

def build_model(input_shape, num_classes):

    inp = Input(shape=input_shape)

    x = Conv2D(32, (3,3), activation="relu")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (3,3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # new block
    x = Conv2D(256, (3,3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # reshape for LSTM
    shape = x.shape
    x = Reshape((shape[1], shape[2]*shape[3]))(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inp, out)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model