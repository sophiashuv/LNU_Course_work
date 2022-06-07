from train_model import *


def model_predict(model, X_test, threshold="no"):
    preds_test = model.predict(X_test, verbose=1)
    if threshold == "no":
        return preds_test
    preds_test = (preds_test > threshold).astype(np.uint8)
    return preds_test


def predict_image(model, IMG_PATH, TEST_PATH, MASK_TEST_PATH, WEIGHTS_PATH, TRESHHOLD):
    X_test, Y_test = read_test_img_mask(IMG_PATH, TEST_PATH, MASK_TEST_PATH)
    model.load_weights(filepath=WEIGHTS_PATH)
    Y_pred = model_predict(model, X_test, TRESHHOLD)
    Y_test, Y_pred = np.squeeze(Y_test), np.squeeze(Y_pred)
    Y_test, Y_pred = np.reshape(Y_test, (1, IMG_HEIGHT, IMG_WIDTH)), np.reshape(Y_pred, (1, IMG_HEIGHT, IMG_WIDTH))
    return np.array([IMG_PATH]), X_test, Y_test, Y_pred


def predict_images(model, TEST_PATH, MASK_TEST_PATH, WEIGHTS_PATH, TRESHHOLD, amount=-1):
    test_ids = get_images_ids(TEST_PATH, amount)
    X_test, Y_test = resizing_train_data(test_ids, TEST_PATH, MASK_TEST_PATH)
    model.load_weights(filepath=WEIGHTS_PATH)
    Y_pred = model_predict(model, X_test, TRESHHOLD)
    n = Y_test.shape[0]
    Y_test, Y_pred = np.reshape(Y_test, (n, IMG_HEIGHT, IMG_WIDTH)), np.reshape(Y_pred, (n, IMG_HEIGHT, IMG_WIDTH))
    return test_ids, X_test, Y_test, Y_pred


def save_predicted_data(architecture):
    if architecture == 'U-Net':
        model = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    elif architecture == 'FCN':
        model = FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    else:
        return

    if task == "1":
        test_ids, X_test, Y_test, Y_pred = predict_image(model, IMG_PATH, TEST_PATH, MASK_TEST_PATH, WEIGHTS_PATH, TRESHHOLD)
    elif task == "2":
        test_ids, X_test, Y_test, Y_pred = predict_images(model, TEST_PATH, MASK_TEST_PATH, WEIGHTS_PATH, TRESHHOLD)
    else:
        return
    save_images(test_ids, Y_pred, SAVE_PATH, FORMAT)
    return test_ids, X_test, Y_test, Y_pred


if __name__ == '__main__':
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    TRESHHOLD = 'no'

    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        default='2',
                        help='Possible: 1 - predict single image, 2 - predict many images')
    parser.add_argument('--architecture', default='FCN', help='Possible: U-Net, FCN')
    parser.add_argument("--TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tes/',
                        help='TEST PATH')
    parser.add_argument("--MASK_TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\2018_01_08_tes/',
                        help='TEST MASK PATH')
    parser.add_argument("--WEIGHTS_PATH", default=r"D:\Azure Repository\LNU_Course_work\FCN_data\FCN_model_epoch=5_valloss=0.1269.h5")
    parser.add_argument("--IMG_PATH",
                        default=r"5F781B80-71AA-4BE3-ABD7-0198659685C7.jpg")
    parser.add_argument("--SAVE_PATH",
                        default=r"D:\Azure Repository\LNU_Course_work\predictions\FCN", help='Path to save folder')
    parser.add_argument("--FORMAT",
                        default=".npy", help='Path to save folder')

    args = parser.parse_args()
    task = args.task
    architecture = args.architecture
    TEST_PATH = args.TEST_PATH
    MASK_TEST_PATH = args.MASK_TEST_PATH
    WEIGHTS_PATH = args.WEIGHTS_PATH
    IMG_PATH = args.IMG_PATH
    SAVE_PATH = args.SAVE_PATH
    FORMAT = args.FORMAT

    save_predicted_data(architecture)
