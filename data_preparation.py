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


def read_predicted_images(TEST_PATH, MASK_TEST_PATH, PRED_PATH, TRESHHOLD, amount=-1):
    test_ids = get_images_ids(TEST_PATH, amount)
    X_test, Y_test = resizing_train_data(test_ids, TEST_PATH, MASK_TEST_PATH)
    Y_pred = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        mask = cv2.imread(PRED_PATH + id_[:-3] + "jpg", 0)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = np.reshape(mask, (IMG_HEIGHT, IMG_WIDTH, 1))
        Y_pred[n] = mask/255
    if TRESHHOLD != "no":
        Y_pred = (Y_pred > TRESHHOLD).astype(np.uint8)
    n = Y_test.shape[0]
    Y_test, Y_pred = np.reshape(Y_test, (n, IMG_HEIGHT, IMG_WIDTH)), np.reshape(Y_pred, (n, IMG_HEIGHT, IMG_WIDTH))
    return test_ids, X_test, Y_test, Y_pred


def read_predicted_image(img_id, TEST_PATH, MASK_TEST_PATH, PRED_PATH, TRESHHOLD):
    X_test, Y_test = read_test_img_mask(img_id, TEST_PATH, MASK_TEST_PATH)
    Y_pred = cv2.imread(PRED_PATH + img_id[:-3] + "jpg", 0)
    Y_pred = cv2.resize(Y_pred, (IMG_WIDTH, IMG_HEIGHT))
    Y_pred = np.reshape(Y_pred, (IMG_HEIGHT, IMG_WIDTH, 1))/255.0
    Y_pred = np.array([Y_pred])
    if TRESHHOLD != "no":
        Y_pred = (Y_pred > TRESHHOLD).astype(np.uint8)
    n = Y_test.shape[0]
    Y_test, Y_pred = np.reshape(Y_test, (n, IMG_HEIGHT, IMG_WIDTH)), np.reshape(Y_pred, (n, IMG_HEIGHT, IMG_WIDTH))
    return np.array([img_id]), X_test, Y_test, Y_pred