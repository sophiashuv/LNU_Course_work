import argparse
import os
from U_Net import *

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3


def get_dataset(tfrecordlist, prepfun, batchsize=-1):
    cpu_count = os.cpu_count()
    dataset = tf.data.TFRecordDataset(tfrecordlist, num_parallel_reads=cpu_count)
    dataset = dataset.map(prepfun, num_parallel_calls=cpu_count)
    dataset = dataset.prefetch(batchsize)
    return dataset


def bytes_to_image(bytes_t, height, width, channels, dtype):
    image_t = tf.image.decode_image(bytes_t, channels=channels)
    image_t = tf.cast(image_t, dtype)
    image_t.set_shape([height, width, channels])
    return image_t


def extract_tfrecord(example_proto):
    features = {
        "url":   tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    url_t = parsed_features["url"]
    image_bytes_t = parsed_features["image"]
    mask_bytes_t = parsed_features["mask"]
    return url_t, image_bytes_t, mask_bytes_t


def make_prepfun(width, height, classes, gray):
    def prepfun(example_proto):
        url_t, image_bytes_t, mask_bytes_t = extract_tfrecord(example_proto)
        image_t = bytes_to_image(image_bytes_t, height, width, 3, tf.uint8)
        mask_t = bytes_to_image(mask_bytes_t, height, width, 1, tf.uint8)
        if gray:
            image_t = tf.image.rgb_to_grayscale(image_t)
            image_t.set_shape((height, width, 1))
        else:
            image_t.set_shape((height, width, 3))
        image_t = tf.cast(image_t, tf.float32)
        if classes > 1:
            mask_t = tf.one_hot(mask_t, classes, axis=2)
        mask_t = tf.cast(mask_t, tf.float32)
        return image_t, mask_t

    return prepfun


def get_sample_count(tfrecords):
    dataset = tf.data.TFRecordDataset(tfrecords)
    count = sum([1 for _ in dataset])
    return count


def get_dataset_from_tfrecord(training_tfrecords,
                 validation_tfrecords,
                 width,
                 height,
                 classes,
                 gray,
                 batchsize,
                 batches_in_epoch=-1):
    if batches_in_epoch <= 0:
        training_image_count = get_sample_count(training_tfrecords)
        tra_steps_per_epoch = training_image_count // batchsize
    else:
        tra_steps_per_epoch = batches_in_epoch
    validation_image_count = get_sample_count(validation_tfrecords)
    val_steps_per_epoch = validation_image_count // batchsize

    training_prepfun = make_prepfun(width, height, classes, gray)
    tra_dataset = get_dataset(training_tfrecords, training_prepfun, batchsize)
    validation_prepfun = make_prepfun(width, height, classes, gray)
    val_dataset = get_dataset(validation_tfrecords, validation_prepfun, batchsize)
    return tra_dataset, val_dataset, tra_steps_per_epoch, val_steps_per_epoch


def create_model(inputs, outputs):
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def model_fit(model, tra_dataset, tra_steps_per_epoch, val_dataset, val_steps_per_epoch):
    # results = model.fit_generator(train_ds, validation_data=validation_ds, steps_per_epoch=25, callbacks=callbacks)

    checkpointsuffix = "_epoch={epoch:d}_valloss={val_loss:.4f}.h5"
    checkpointfile = os.path.join(SAVE_PATH, "u-net_model" + checkpointsuffix)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpointfile, verbose=1, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

    results = model.fit(tra_dataset,validation_data=val_dataset, epochs=25, steps_per_epoch=tra_steps_per_epoch, validation_steps=val_steps_per_epoch, callbacks=callbacks)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TRAIN_PATH', default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\train/', help='TRAIN PATH')
    parser.add_argument("--MASK_PATH", default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\train/', help='MASK MATH')
    parser.add_argument("--TEST_PATH", default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\test/', help='TEST PATH')
    parser.add_argument("--MASK_TEST_PATH", default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\test/', help='MASK TEST MATH')
    parser.add_argument("--SAVE_PATH", default=r'D:\Azure Repository\LNU_Course_work\data/', help='SAVE_PATH')

    args = parser.parse_args()
    TRAIN_PATH = args.TRAIN_PATH
    MASK_PATH = args.MASK_PATH
    TEST_PATH = args.TEST_PATH
    MASK_TEST_PATH = args.MASK_TEST_PATH
    SAVE_PATH = args.SAVE_PATH

    tra_dataset, val_dataset, tra_steps_per_epoch, val_steps_per_epoch = get_dataset_from_tfrecord(r"D:\Azure Repository\LNU_Course_work\try_train.tfrecord",
                                                                                                   r"D:\Azure Repository\LNU_Course_work\try_validation.tfrecord",
                              IMG_WIDTH,
                              IMG_WIDTH,
                              1,
                              False,
                              25,
                              -1)
    inputs, outputs = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    model = create_model(inputs, outputs)
    model_fit(model, tra_dataset, tra_steps_per_epoch, val_dataset, val_steps_per_epoch)
