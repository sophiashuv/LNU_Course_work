import numpy as np
import cv2
from tqdm import tqdm
import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
batch_size = 4


def to_bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def get_train_validation_data_img_mask_files(TRAIN_PATH, MASK_PATH, train=0.9):
    train_ids = next(os.walk(TRAIN_PATH))[2][:-1]
    mask_ids = next(os.walk(MASK_PATH))[2][:-1]
    train_ids = np.array(train_ids)
    mask_ids = np.array(mask_ids)
    files = np.stack((train_ids, mask_ids)).T
    train_len = int(files.shape[0] * train)
    return files[:, :train_len], files[:, train_len:]


def image_masks_to_tfrecord(imagedir, maskdir, tfrecordfile, image_mask_pairs):
    writer = tf.io.TFRecordWriter(tfrecordfile)
    for i, (imgfile, maskfile) in tqdm(enumerate(image_mask_pairs), total=image_mask_pairs.shape[0]):
        imgpath = imagedir + imgfile
        maskpath = maskdir + maskfile
        image_bgr = cv2.imread(imgpath)
        mask = cv2.imread(maskpath)
        _, image_jpeg = cv2.imencode(".jpg", image_bgr)
        _, mask_png = cv2.imencode(".png", mask)
        feature = {
            "url":   to_bytes_feature(imgpath.encode("utf-8")),
            "image": to_bytes_feature(image_jpeg.tobytes()),
            "mask":  to_bytes_feature(mask_png.tobytes())
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TRAIN_PATH', default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\train\tra/', help='TRAIN PATH')
    parser.add_argument("--MASK_PATH", default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\train\tra/', help='MASK MATH')


    args = parser.parse_args()
    TRAIN_PATH = args.TRAIN_PATH
    MASK_PATH = args.MASK_PATH
    train_image_mask_pairs, validation_image_mask_pairs = get_train_validation_data_img_mask_files(TRAIN_PATH, MASK_PATH)
    image_masks_to_tfrecord(TRAIN_PATH, MASK_PATH, r"D:\Azure Repository\LNU_Course_work\try_validation.tfrecord", train_image_mask_pairs)
    image_masks_to_tfrecord(TRAIN_PATH, MASK_PATH, r"D:\Azure Repository\LNU_Course_work\try_train.tfrecord",
                           train_image_mask_pairs)
    # raw_dataset = tf.data.TFRecordDataset(r"D:\Azure Repository\LNU_Course_work\try.tfrecord")
    #
    # for raw_record in raw_dataset.take(2):
    #     example = tf.train.Example()
    #     example.ParseFromString(raw_record.numpy())
    #     print(example)