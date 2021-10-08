# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:28:10 2021

@author: safayet_khan
"""

import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from siamese_augmentation import image_augmentation

random.seed(a=42)


def load_data_npy(data_path, image_size):
    """
    Load images in numpy array. Images are load as RGB format.
    Input: Main data directory and image size in 2D.
    Output: X and Y as numpy array.
    """
    directory_list = os.listdir(path=data_path)
    directory_list.sort()
    x_data = list()
    y_data = list()

    for directory in directory_list:
        class_path = os.path.join(data_path, directory)
        image_names = os.listdir(path=class_path)
        image_names.sort()
        for image_name in image_names:
            image_array = cv2.imread(
                os.path.join(class_path, image_name), cv2.IMREAD_COLOR
            )
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            image_array = cv2.resize(image_array, image_size)
            x_data.append(image_array)
            y_data.append(int(directory))

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


def plot_duplet(example):
    """
    Plot a random example of duplet.
    """
    plt.figure(figsize=(4, 2))
    for i in range(2):
        plt.subplot(1, 2, 1 + i)
        plt.imshow(example[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def positive_duplet(x_value, y_value, augmentation):
    """
    Create a random positive duplet from data.
    If augmentation is True, then image augmentation is applied
    for the both images of negative duplet.
    """
    random_index = random.randint(0, x_value.shape[0] - 1)

    if augmentation is True:
        x_anchor = image_augmentation(x_value[random_index])
    else:
        x_anchor = x_value[random_index]

    y_anchor = y_value[random_index]
    pos_index = np.squeeze(np.where(y_value == y_anchor))

    if augmentation is True:
        x_positive = image_augmentation(
            x_value[pos_index[random.randint(0, len(pos_index) - 1)]]
        )
    else:
        x_positive = x_value[pos_index[random.randint(0, len(pos_index) - 1)]]

    return [x_anchor, x_positive], 1.0


def negative_duplet(x_value, y_value, augmentation):
    """
    Create a random negative duplet from data.
    If augmentation is True, then image augmentation is applied
    for the both images of negative duplet.
    """
    random_index = random.randint(0, x_value.shape[0] - 1)

    if augmentation is True:
        x_anchor = image_augmentation(x_value[random_index])
    else:
        x_anchor = x_value[random_index]

    y_anchor = y_value[random_index]
    neg_index = np.squeeze(np.where(y_value != y_anchor))

    if augmentation is True:
        x_negative = image_augmentation(
            x_value[neg_index[random.randint(0, len(neg_index) - 1)]]
        )
    else:
        x_negative = x_value[neg_index[random.randint(0, len(neg_index) - 1)]]

    return [x_anchor, x_negative], 0.0


def create_batch(batch_size, x_value, y_value, augmentation):
    """
    Create a batch of duplets given fixed batch size
    If augmentation is True, then image augmentation is applied
    for the full a batch of images.
    """
    x_batch = list()
    y_batch = list()
    for _ in range(0, batch_size):
        random_number = random.randint(0, 1)
        if random_number == 0:
            x_list, index = positive_duplet(x_value, y_value, augmentation)
        else:
            x_list, index = negative_duplet(x_value, y_value, augmentation)

        x_batch.append(x_list)
        y_batch.append(index)
    return np.array(x_batch) / 255.0, np.array(y_batch)


def yield_data(data_path, image_size, batch_size, sample_size, augmentation):

    """
    First image folders are randomly sampled.
    Then sampled folder images are loaded in numpy array.
    Images are load as RGB format.
    From loaded data it will generate batchs of duplet
    data continuously. Used while training.

    batch_size: how many batch of duplet in each yield.
    sample_size: how many folders have to randomly sample each time.
    If augmentation is True, then image augmentation is applied
    for the full every batchs of images.

    """
    directory_list = os.listdir(path=data_path)
    directory_list.sort()

    while True:
        sample_directory_list = random.sample(directory_list, k=sample_size)
        sample_directory_list.sort()
        x_data = list()
        y_data = list()

        for sample_directory in sample_directory_list:
            class_path = os.path.join(data_path, sample_directory)
            image_names = os.listdir(path=class_path)
            image_names.sort()
            for image_name in image_names:
                image_array = cv2.imread(
                    os.path.join(class_path, image_name), cv2.IMREAD_COLOR
                )
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                image_array = cv2.resize(image_array, image_size)
                x_data.append(image_array)
                y_data.append(int(sample_directory))

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        x_batch, y_batch = create_batch(batch_size, x_data, y_data, augmentation)
        yield [x_batch[:, 0, :], x_batch[:, 1, :]], y_batch
