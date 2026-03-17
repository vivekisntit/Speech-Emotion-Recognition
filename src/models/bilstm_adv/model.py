from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D,
    BatchNormalization, Dropout,
    Reshape, Bidirectional, LSTM,
    Dense, Multiply, Softmax, GlobalAveragePooling1D
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

    x = Conv2D(256, (3,3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    
    shape = x.shape
    x = Reshape((shape[1], shape[2]*shape[3]))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    attention = Dense(1)(x)
    attention = Softmax(axis=1)(attention)
    
    x = Multiply()([x, attention])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.6)(x)
    
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model