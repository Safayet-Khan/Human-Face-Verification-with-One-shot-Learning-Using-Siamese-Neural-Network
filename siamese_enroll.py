# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 11:19:46 2021

@author: safayet_khan
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from siamese_function import plot_duplet

tf.config.experimental_run_functions_eagerly(True)


IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE_2D = (IMAGE_SIZE, IMAGE_SIZE)
IMAGE_SIZE_3D = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

PROJECT_DIRECTORY = "C:/Users/safayet_khan/Desktop/siamese_network/"
ENROLL_DIRCTORY = os.path.join(PROJECT_DIRECTORY, "siamese_enroll_dataset")
INPUT_DATA_DIRCTORY = os.path.join(PROJECT_DIRECTORY, "siamese_input_image")

MODEL_PATH = os.path.join(
    PROJECT_DIRECTORY, "siamese_pretrained_model", "saimese_model.h5"
)
siamese_net = load_model(MODEL_PATH)
siamese_net.summary()


directory_list = os.listdir(path=ENROLL_DIRCTORY)
directory_list.sort()
x_data = list()
y_data = list()

for directory in directory_list:
    class_path = os.path.join(ENROLL_DIRCTORY, directory)
    image_names = os.listdir(path=class_path)
    for image_name in image_names:
        # fmt: off
        image_array = cv2.imread(os.path.join(class_path, image_name),
                                cv2.IMREAD_COLOR)
        # fmt: on
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_array = cv2.resize(image_array, IMAGE_SIZE_2D)
        x_data.append(image_array)
        y_data.append(int(directory))

x_test_enroll = np.array(x_data)
y_test_enroll = np.array(y_data)

print(x_test_enroll.shape)
print(y_test_enroll.shape)


x_test_input = list()
image_names = os.listdir(path=INPUT_DATA_DIRCTORY)

for image_name in image_names:
    # fmt: off
    image_array = cv2.imread(os.path.join(INPUT_DATA_DIRCTORY, image_name),
                            cv2.IMREAD_COLOR)
    # fmt: on
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_array = cv2.resize(image_array, IMAGE_SIZE_2D)
    x_test_input.append(image_array)

x_test_input = np.array(x_test_input)
print(x_test_input.shape)


def test_duplets(total_enrolls, input_image):
    """
    Create test duplets
    """
    dupelt_batch = list()
    for enroll in total_enrolls:
        dupelt_batch.append([enroll, input_image[0]])
    return np.array(dupelt_batch) / 255.0


x_enroll_input_batch = test_duplets(x_test_enroll, x_test_input)
print(x_enroll_input_batch.shape)


y_pred = siamese_net.predict(
    [x_enroll_input_batch[:, 1, :], x_enroll_input_batch[:, 0, :]]
)
if np.any(y_pred >= 0.50):
    location = np.where(y_pred == y_pred.max())[0]
    plot_duplet(
        x_enroll_input_batch[location, :].reshape(
            2, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS
        )
    )
else:
    print("Not in test set")
