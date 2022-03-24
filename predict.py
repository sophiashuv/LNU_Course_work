import random
import matplotlib.pyplot as plt

from train_model import *


def model_predict(model, treshhold=0.5):
    preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
    preds_test = model.predict(X_test, verbose=1)
    preds_train_t = (preds_train > treshhold).astype(np.uint8)
    preds_val_t = (preds_val > treshhold).astype(np.uint8)
    preds_test_t = (preds_test > treshhold).astype(np.uint8)
    # Perform a sanity check on some random training samples
    ix = random.randint(0, len(preds_train_t))

    plot2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.imshow(X_train[ix])
    ax2.imshow(np.squeeze(Y_train[ix]))
    ax3.imshow(np.squeeze(preds_train_t[ix]))
    ax1.set_title("X_train")
    ax2.set_title("Y_train")
    ax3.set_title("preds_train_t")
    plt.show()

    # Perform a sanity check on some random validation samples
    ix = random.randint(0, len(preds_val_t))
    plot2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.imshow(X_train[int(X_train.shape[0] * 0.9):][ix])
    ax2.imshow(np.squeeze(Y_train[int(Y_train.shape[0] * 0.9):][ix]))
    ax3.imshow(np.squeeze(preds_val_t[ix]))
    ax1.set_title("X_validation")
    ax2.set_title("Y_validation")
    ax3.set_title("preds_val_t")
    plt.show()

    # Perform a sanity check on some random test samples
    ix = random.randint(0, len(preds_test_t))
    plot2, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    ax1.imshow(X_test[ix])
    ax2.imshow(np.squeeze(preds_test_t[ix]))
    ax1.set_title("X_test")
    ax2.set_title("preds_test_t")
    plt.show()


if __name__ == '__main__':
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    IMG_CHANNELS = 3

    TRESHHOLD = 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument('--TRAIN_PATH',
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tra/',
                        help='TRAIN PATH')
    parser.add_argument("--MASK_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\2018_01_08_tra\2018_01_08_tra/',
                        help='MASK MATH')
    parser.add_argument("--TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tes/',
                        help='TEST PATH')
    parser.add_argument("--SAVE_PATH", default=r'D:\Azure Repository\LNU_Course_work\data', help='SAVE_PATH')
    parser.add_argument("--WEIGHTS_PATH", default=r"D:\Azure Repository\LNU_Course_work\FCN_data\FCN_model_epoch=1_valloss=0.7011.h5")
    args = parser.parse_args()
    TRAIN_PATH = args.TRAIN_PATH
    MASK_PATH = args.MASK_PATH
    TEST_PATH = args.TEST_PATH
    SAVE_PATH = args.SAVE_PATH
    WEIGHTS_PATH = args.WEIGHTS_PATH

    train_ids = next(os.walk(TRAIN_PATH))[2][:-1]
    mask_ids = next(os.walk(MASK_PATH))[2][:-1]
    test_ids = next(os.walk(TEST_PATH))[2][:-1]

    X_train, Y_train = resizing_train_data(train_ids, TRAIN_PATH, MASK_PATH)
    X_test = resizing_test_data(test_ids, TEST_PATH)

    model = FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

    model.load_weights(filepath=WEIGHTS_PATH)
    model_predict(model, TRESHHOLD)
