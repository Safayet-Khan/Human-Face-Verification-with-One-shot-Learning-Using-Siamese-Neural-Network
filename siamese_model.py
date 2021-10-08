# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:43:01 2021

@author: safayet_khan
"""

import math
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Lambda, Conv2D
from tensorflow.keras.layers import MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler, TerminateOnNaN
from siamese_function import create_batch, load_data_npy, yield_data
from siamese_function import plot_duplet, positive_duplet, negative_duplet

tf.config.experimental_run_functions_eagerly(True)


IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE_2D = (IMAGE_SIZE, IMAGE_SIZE)
IMAGE_SIZE_3D = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

PROJECT_DIRECTORY = "C:/Users/safayet_khan/Desktop/siamese_network/"
DATA_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "siamese_data")
TRAIN_DIRECTORY = os.path.join(DATA_DIRECTORY, "train")
VALID_DIRECTORY = os.path.join(DATA_DIRECTORY, "valid")
TEST_DIRECTORY = os.path.join(DATA_DIRECTORY, "test")


x_valid, y_valid = load_data_npy(VALID_DIRECTORY, IMAGE_SIZE_2D)
print(x_valid.shape)
print(y_valid.shape)

x_test, y_test = load_data_npy(TEST_DIRECTORY, IMAGE_SIZE_2D)
print(x_test.shape)
print(y_test.shape)


plot_duplet([x_valid[0], x_valid[1]])

example, _ = positive_duplet(x_valid, y_valid, augmentation=True)
plot_duplet(example)

example, _ = negative_duplet(x_valid, y_valid, augmentation=False)
plot_duplet(example)


def feature_learning_network(input_shape):
    """
    Base of the siamese neural network
    """
    feature_model = Sequential()
    feature_model.add(Input(shape=IMAGE_SIZE_3D))

    feature_model.add(
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")
    )
    feature_model.add(BatchNormalization())
    feature_model.add(
        Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")
    )
    feature_model.add(BatchNormalization())
    feature_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    feature_model.add(
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
    )
    feature_model.add(BatchNormalization())
    feature_model.add(
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
    )
    feature_model.add(BatchNormalization())
    feature_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    feature_model.add(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
    )
    feature_model.add(BatchNormalization())
    feature_model.add(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
    )
    feature_model.add(BatchNormalization())
    feature_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    feature_model.add(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
    )
    feature_model.add(BatchNormalization())
    feature_model.add(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
    )
    feature_model.add(BatchNormalization())
    feature_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    feature_model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
    )
    feature_model.add(BatchNormalization())
    feature_model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
    )
    feature_model.add(BatchNormalization())
    feature_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    feature_model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
    )
    feature_model.add(BatchNormalization())
    feature_model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
    )
    feature_model.add(BatchNormalization())
    feature_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    feature_model.add(Flatten())
    feature_model.add(Dense(units=512, activation="sigmoid"))
    feature_model.summary()

    return feature_model


def classification_network(feature_model, input_shape):
    """
    Lower part of the siamese neural network
    """
    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)

    embed_left = feature_model(left_input)
    embed_right = feature_model(right_input)

    # Add a customized layer to compute the absolute difference
    l1_layer = Lambda(lambda tensors: tf.math.abs(tensors[0] - tensors[1]))
    l1_distance = l1_layer([embed_left, embed_right])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    network_output = Dense(units=1, activation="sigmoid")(l1_distance)

    network_input = [left_input, right_input]
    siamese_net = Model(inputs=network_input, outputs=network_output)
    siamese_net.summary()

    return siamese_net


VALIDATION_ACCURACY_THRESHOLD = 0.9999


class MyCallback(Callback):
    """
    Stop training after val_accuracy reached a certain number.
    """

    def on_epoch_end(self, epoch, logs=None):
        if logs.get("val_accuracy") > VALIDATION_ACCURACY_THRESHOLD:
            print("\nCOMPLETED!!!")
            self.model.stop_training = True


callbacks = MyCallback()


CHECKPOINT_PATH = os.path.join(PROJECT_DIRECTORY, "checkpoint_folder")
if not os.path.exists(path=CHECKPOINT_PATH):
    os.mkdir(path=CHECKPOINT_PATH)

CHECKPOINT_FILEPATH = os.path.join(CHECKPOINT_PATH, "saimese_model.h5")
checkpoint_callback = ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    monitor="val_loss",
    mode="min",
    verbose=1,
    save_weights_only=False,
    save_best_only=True,
    save_freq="epoch",
)


INITIAL_LEARNING_RATE = 0.02


def lr_step_decay(epoch):
    """
    Reduce the learning rate by a certain percentage
    after a certain number of epochs.
    """
    drop_rate = 0.25
    epochs_drop = 15.0
    return INITIAL_LEARNING_RATE * math.pow(
        (1 - drop_rate), math.floor(epoch / epochs_drop)
    )


lr_callback = LearningRateScheduler(schedule=lr_step_decay, verbose=1)


CSVLOGGER_PATH = os.path.join(PROJECT_DIRECTORY, "log_folder")
if not os.path.exists(path=CSVLOGGER_PATH):
    os.mkdir(path=CSVLOGGER_PATH)

CSVLOGGER_FILEPATH = os.path.join(CSVLOGGER_PATH, "saimese_log.csv")
# fmt: off
CSVLOGGER_callback = CSVLogger(filename=CSVLOGGER_FILEPATH,
                               separator=",", append=False)
# fmt:on

Terminate_callback = TerminateOnNaN()


EPSILON = 0.1

base_model = feature_learning_network(IMAGE_SIZE_3D)
siamese_net = classification_network(base_model, IMAGE_SIZE_3D)
siamese_net.compile(
    optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE, epsilon=EPSILON),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


VALIDATION_BATCH = 1800

x_valid_batch, y_valid_batch = create_batch(
    VALIDATION_BATCH, x_valid, y_valid, augmentation=False
)

del x_valid
del y_valid

SAMPLE_SIZE = 20
BATCH_SIZE = 64
EPOCHS = 225
STEPS_PER_EPOCH = 60

siamese_net.fit(
    yield_data(
        data_path=TRAIN_DIRECTORY,
        image_size=IMAGE_SIZE_2D,
        batch_size=BATCH_SIZE,
        sample_size=SAMPLE_SIZE,
        augmentation=True,
    ),
    verbose=True,
    steps_per_epoch=STEPS_PER_EPOCH,
    shuffle=False,
    epochs=EPOCHS,
    # fmt:off
    validation_data=([x_valid_batch[:, 0, :], x_valid_batch[:, 1, :]],
                     y_valid_batch),
    # fmt:on
    callbacks=[
        callbacks,
        checkpoint_callback,
        lr_callback,
        CSVLOGGER_callback,
        Terminate_callback,
    ],
)
