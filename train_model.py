import argparse
import os

from data_preparation import *
from models.U_Net import *
from models.FCN import *

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3


def create_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def model_fit(model, X_train, Y_train, SAVE_PATH, log_dir):
    checkpointsuffix = "_epoch={epoch:d}_valloss={val_loss:.4f}.h5"
    checkpointfile = os.path.join(SAVE_PATH, "FCN_model" + checkpointsuffix)

    callbacks = [
            tf.keras.callbacks.ModelCheckpoint(checkpointfile, verbose=1, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)
    return results


def train_model(architecture, TRAIN_PATH, MASK_PATH, SAVE_PATH):
    if architecture == 'U-Net':
        model = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    elif architecture == 'FCN':
        model = FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    else:
        return
    train_ids = get_images_ids(TRAIN_PATH)

    X_train, Y_train = resizing_train_data(train_ids, TRAIN_PATH, MASK_PATH)

    model = create_model(model)
    model_fit(model, X_train, Y_train, SAVE_PATH, 'logs_' + architecture)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture',
                        default='FCN',
                        help='Possible: U-Net, FCN')
    parser.add_argument('--TRAIN_PATH',
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tra/',
                        help='TRAIN PATH')
    parser.add_argument("--MASK_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\2018_01_08_tra\2018_01_08_tra/',
                        help='MASK MATH')
    parser.add_argument("--SAVE_PATH",
                        default=r'D:\Azure Repository\LNU_Course_work\FCN_data/',
                        help='SAVE_PATH')

    args = parser.parse_args()

    architecture = args.architecture
    TRAIN_PATH = args.TRAIN_PATH
    MASK_PATH = args.MASK_PATH
    SAVE_PATH = args.SAVE_PATH

    train_model(architecture, TRAIN_PATH, MASK_PATH, SAVE_PATH)
