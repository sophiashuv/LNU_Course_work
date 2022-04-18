import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import os

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3


def get_images_ids(path, amount=-1):
    train_ids = next(os.walk(path))[2][:-1][:amount]
    return train_ids


def resizing_train_data(train_ids, TRAIN_PATH, MASK_PATH):
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = cv2.imread(path)[:, :, :IMG_CHANNELS]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        X_train[n] = img
        mask = cv2.imread(MASK_PATH + id_[:-3] + "png", 0)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = np.reshape(mask, (IMG_HEIGHT, IMG_WIDTH, 1))
        Y_train[n] = mask
    return X_train, Y_train


def read_test_img_mask(img_id, TEST_PATH, MASK_TEST_PATH):
    img = cv2.imread(TEST_PATH + img_id)[:, :, :IMG_CHANNELS]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    mask = cv2.imread(MASK_TEST_PATH + img_id[:-3] + "png", 0) / 255
    mask_small = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
    mask_small = np.reshape(mask_small, (IMG_HEIGHT, IMG_WIDTH, 1))
    return np.array([img_small]), np.array([mask_small])


def save_images(ids, images, save_path):
    if np.max(images) <= 1:
        images = images * 255
    for img, id_ in zip(images, ids):
        cv2.imwrite(os.path.join(save_path, id_), img)

