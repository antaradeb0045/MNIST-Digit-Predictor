# =========================
# TRAIN AND SAVE MODEL
# =========================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

# LOAD DATA
train = pd.read_csv(r"C:\Users\antar\OneDrive\Desktop\DM_A\DM_A\Data resource\train.csv\train.csv")

# PREPARE DATA
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)

# NORMALIZE
X_train = X_train / 255.0

# RESHAPE
X_train = X_train.values.reshape(-1, 28, 28, 1)

# ONE HOT ENCODING
Y_train = to_categorical(Y_train, num_classes=10)

# DATA AUGMENTATION
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(X_train)

# BUILD MODEL
def build_model():
    model = Sequential()

    model.add(Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, 4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# LEARNING RATE
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# TRAIN MODEL (single model for simplicity)
model = build_model()

X_train2, X_val2, Y_train2, Y_val2 = train_test_split(
    X_train, Y_train, test_size=0.1
)

history = model.fit(
    datagen.flow(X_train2, Y_train2, batch_size=64),
    epochs=20,
    steps_per_epoch=X_train2.shape[0] // 64,
    validation_data=(X_val2, Y_val2),
    callbacks=[annealer],
    verbose=1
)

# SAVE MODEL
model.save("mnist_cnn_model.h5")

print("✅ Model saved as mnist_cnn_model.h5")