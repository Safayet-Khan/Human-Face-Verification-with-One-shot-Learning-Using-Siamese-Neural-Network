# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 11:19:46 2021

@author: safayet_khan
"""

import os
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
from siamese_function import create_batch, load_data_npy

tf.config.experimental_run_functions_eagerly(True)


IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE_2D = (IMAGE_SIZE, IMAGE_SIZE)
IMAGE_SIZE_3D = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

PROJECT_DIRECTORY = "C:/Users/safayet_khan/Desktop/siamese_network/"
DATA_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "siamese_data")
TEST_DIRECTORY = os.path.join(DATA_DIRECTORY, "test")
MODEL_PATH = os.path.join(
    PROJECT_DIRECTORY, "siamese_pretrained_model", "saimese_model.h5"
)


x_test, y_test = load_data_npy(TEST_DIRECTORY, IMAGE_SIZE_2D)
print(x_test.shape)
print(y_test.shape)


TEST_BATCH = 1500

x_test_batch, y_test_batch = create_batch(
    TEST_BATCH, x_test, y_test, augmentation=False
)

del x_test
del y_test

siamese_net = load_model(MODEL_PATH)
siamese_net.summary()

y_pred = siamese_net.predict([x_test_batch[:, 0, :], x_test_batch[:, 1, :]])
y_pred = (y_pred >= 0.5).astype(int)


# Calculate the accuracy, precision, recall, F1 score
acc_score = round(accuracy_score(y_test_batch, y_pred), 2)
print("Accuracy:", acc_score)

pre_score = round(precision_score(y_test_batch, y_pred, average="weighted"), 2)
print("Precision:", pre_score)

rec_score = round(recall_score(y_test_batch, y_pred, average="weighted"), 2)
print("Recall:", rec_score)

F1_score = round(f1_score(y_test_batch, y_pred, average="weighted"), 2)
print("F1:", F1_score)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(y_test_batch, y_pred)
name_label = ["False", "True"]
plt.figure(figsize=(8, 8))
confusion_heatmap = sns.heatmap(
    confusion_mat,
    annot=True,
    linewidths=0.5,
    xticklabels=name_label,
    yticklabels=name_label,
    fmt=".0f",
    square=True,
    cmap="Blues_r",
)
confusion_heatmap.set(
    title="Accuracy Score: {}".format(acc_score),
    xlabel="Predicted label",
    ylabel="Actual label",
)
plt.show()
