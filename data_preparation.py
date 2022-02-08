import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import matplotlib.pyplot as plt
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
batch_size = 4


def resizing_train_data(train_ids, TRAIN_PATH, MASK_PATH):
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = cv2.imread(path)[:, :, :IMG_CHANNELS]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        X_train[n] = img
        mask = cv2.imread(MASK_PATH + id_[:-3] + "png", 0)
        mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        mask = np.reshape(mask, (IMG_HEIGHT, IMG_WIDTH, 1))
        Y_train[n] = mask
    return X_train, Y_train


def resizing_test_data(test_ids, TEST_PATH):
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = cv2.imread(path)[:, :, :IMG_CHANNELS]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        X_test[n] = img
    return X_test


def my_image_mask_generator(image_data_generator, mask_data_generator):
    train_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

# model = ... # Define your model here
# # Compile your model here
# model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#
# # Train your model here
# model.fit_generator(my_generator,...)

def get_train_data(TRAIN_PATH, MASK_PATH):
    train_images_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TRAIN_PATH,
        label_mode=None,
        validation_split=0.2,
        subset="training",
        seed=123,
        shuffle=False,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size)

    train_mask_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=MASK_PATH,
        label_mode=None,
        validation_split=0.2,
        subset="training",
        seed=123,
        shuffle=False,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size)

    train_generator = my_image_mask_generator(train_images_ds, train_mask_ds)
    return train_generator


def get_validation_data(TRAIN_PATH, MASK_PATH):
    validation_images_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TRAIN_PATH,
        label_mode=None,
        validation_split=0.2,
        subset="validation",
        seed=123,
        shuffle=False,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size)

    validation_mask_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=MASK_PATH,
        label_mode=None,
        validation_split=0.2,
        subset="validation",
        seed=123,
        shuffle=False,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size)

    validation_generator = my_image_mask_generator(validation_images_ds, validation_mask_ds)
    return validation_generator


def get_test_data(TEST_PATH, MASK_TEST_PATH):
    test_images_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TEST_PATH,
        label_mode=None,
        subset="test",
        seed=123,
        shuffle=False,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size)

    test_mask_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=MASK_TEST_PATH,
        label_mode=None,
        subset="test",
        seed=123,
        shuffle=False,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size)

    test_generator = my_image_mask_generator(test_images_ds, test_mask_ds)
    return test_generator

