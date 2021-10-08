# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 16:05:28 2021

@author: safayet_khan
"""

import random
import numpy as np
import cv2
from skimage.transform import rotate
from skimage.exposure import adjust_gamma
from skimage.util import random_noise

random.seed(a=42)

# Variable for image augmentation
ANGLE_LIMIT = 20
GAMMA_LIMIT = 0.25
A_MIN = 0
A_MAX = 255


def image_augmentation(image_array):
    """
    Input: Take image as a numpy array. The image is loaded in RGB format.
    Output: Return a numpy array with random image augmentation.
    """
    random_transform = random.randint(0, 5)
    if random_transform == 0:
        image_transform = np.fliplr(image_array)
    elif random_transform == 1:
        image_transform = np.flipud(image_array)
    elif random_transform == 2:
        image_transform = image_array
    else:
        random_angle = random.randint(-ANGLE_LIMIT, ANGLE_LIMIT)
        image_transform = rotate(image=image_array, angle=random_angle, mode="edge")

    random_exposure = random.randint(0, 4)
    if random_exposure == 0:
        random_gamma = round(random.uniform(1 - GAMMA_LIMIT, 1 + GAMMA_LIMIT), 2)
        image_exposure = np.clip(
            adjust_gamma(image=image_transform, gamma=random_gamma),
            a_min=A_MIN,
            a_max=A_MAX,
        )
    elif random_exposure == 1:
        ksize = random.randint(2, 3)
        image_exposure = np.clip(
            cv2.blur(image_transform, (ksize, ksize)), a_min=A_MIN, a_max=A_MAX
        )
    elif random_exposure == 2:
        random_sp_amount = round(random.uniform(0.01, 0.025), 3)
        image_exposure = np.clip(
            random_noise(image=image_transform, mode="s&p", amount=random_sp_amount),
            a_min=A_MIN,
            a_max=A_MAX,
        )
    else:
        image_exposure = image_transform

    return image_exposure
