import numpy as np
import cv2
from tqdm import tqdm

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

BIG_IMG_WIDTH = 5456
BIG_IMG_HEIGHT = 3632


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


def resizing_test_masks(test_ids, MASK_PATH):
    X_test_masks = np.zeros((len(test_ids), BIG_IMG_HEIGHT, BIG_IMG_WIDTH, 1),  dtype=np.uint8)
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        mask_path = MASK_PATH + id_[:-3] + "png"
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (BIG_IMG_WIDTH, BIG_IMG_HEIGHT))
        mask = np.reshape(mask, (BIG_IMG_HEIGHT, BIG_IMG_WIDTH, 1))
        X_test_masks[n] = mask
    return X_test_masks


def resizing_test_data(test_ids, TEST_PATH):
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = cv2.imread(path)[:, :, :IMG_CHANNELS]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        X_test[n] = img
    return X_test

