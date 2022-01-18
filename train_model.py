import argparse
import os

from data_preparation import *
from U_Net import *

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3


def create_model(inputs, outputs):
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def model_fit(model, X_train, Y_train):
    checkpointsuffix = "_epoch={epoch:d}_valloss={val_loss:.4f}.h5"
    checkpointfile = os.path.join(SAVE_PATH, "u-net_model" + checkpointsuffix)

    callbacks = [
            tf.keras.callbacks.ModelCheckpoint(checkpointfile, verbose=1, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs')]

    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TRAIN_PATH', default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tra/', help='TRAIN PATH')
    parser.add_argument("--MASK_PATH", default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\2018_01_08_tra\2018_01_08_tra/', help='MASK MATH')
    parser.add_argument("--TEST_PATH", default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tes/', help='TEST PATH')
    parser.add_argument("--SAVE_PATH", default=r'D:\Azure Repository\LNU_Course_work\data/', help='SAVE_PATH')

    args = parser.parse_args()
    TRAIN_PATH = args.TRAIN_PATH
    MASK_PATH = args.MASK_PATH
    TEST_PATH = args.TEST_PATH
    SAVE_PATH = args.SAVE_PATH

    train_ids = next(os.walk(TRAIN_PATH))[2][:-1]
    mask_ids = next(os.walk(MASK_PATH))[2][:-1]
    test_ids = next(os.walk(TEST_PATH))[2][:-1]

    X_train, Y_train = resizing_train_data(train_ids, TRAIN_PATH, MASK_PATH)
    X_test = resizing_test_data(test_ids, TEST_PATH)

    inputs, outputs = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    model = create_model(inputs, outputs)
    model_fit(model, X_train, Y_train)
