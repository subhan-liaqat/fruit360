"""Section 4: fixed Keras CNN architecture for Fruits-360."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape: tuple[int, int, int] = (100, 100, 3), num_classes: int = 10) -> keras.Model:
    """Two conv (8 filters, 3x3), max pool 2x2, two dense (16), softmax output."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(8, (3, 3), padding="same", activation="relu"),
            layers.Conv2D(8, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="fruit360_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
