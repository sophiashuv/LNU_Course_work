import random
import matplotlib.pyplot as plt

from train_model import *
TRESHHOLD = 0.5


def model_predict(model):
    preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
    preds_test = model.predict(X_test, verbose=1)
    preds_train_t = (preds_train > TRESHHOLD).astype(np.uint8)
    preds_val_t = (preds_val > TRESHHOLD).astype(np.uint8)
    preds_test_t = (preds_test > TRESHHOLD).astype(np.uint8)
    # Perform a sanity check on some random training samples
    ix = random.randint(0, len(preds_train_t))
    plt.imshow(X_train[ix])
    plt.show()
    plt.imshow(np.squeeze(Y_train[ix]))
    plt.show()
    plt.imshow(np.squeeze(preds_train_t[ix]))
    plt.show()

    # Perform a sanity check on some random validation samples
    ix = random.randint(0, len(preds_val_t))
    plt.imshow(X_train[int(X_train.shape[0] * 0.9):][ix])
    plt.show()
    plt.imshow(np.squeeze(Y_train[int(Y_train.shape[0] * 0.9):][ix]))
    plt.show()
    plt.imshow(np.squeeze(preds_val_t[ix]))
    plt.show()

    # Perform a sanity check on some random test samples
    ix = random.randint(0, len(preds_test_t))
    plt.imshow(X_train[int(X_train.shape[0] * 0.9):][ix])
    plt.show()
    plt.imshow(np.squeeze(preds_val_t[ix]))
    plt.show()


if __name__ == '__main__':
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

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

    model.load_weights(filepath=r"D:\Azure Repository\LNU_Course_work\data\u-net_model_epoch=7_valloss=0.2177.h5")
    model_predict(model)