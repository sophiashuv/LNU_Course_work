from keras import backend as K
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from train_model import *


def jacard_coef(y_true, y_pred):
    intersection = np.sum(y_pred * y_true)
    union = np.sum(y_pred) + np.sum(y_true) - intersection
    if union == 0:
        return 0
    return intersection / union


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def model_predict(model, X_test, Y_test, test_ids, metrics, WEIGHTS_PATH, treshhold=0.5):
    preds_test = model.predict(X_test, verbose=1)
    preds_test_t = (preds_test > treshhold).astype(np.uint8)
    preds_test_t_big = np.array(
        [np.reshape(
            cv2.resize(mask, (BIG_IMG_WIDTH, BIG_IMG_HEIGHT)),
            (BIG_IMG_HEIGHT, BIG_IMG_WIDTH, 1))
            for mask in preds_test_t])
    for predicted_mask, groundtruth_mask, id_ in zip(Y_test, preds_test_t_big, test_ids):
        iou = jacard_coef(predicted_mask, groundtruth_mask)
        dice = 1.0 - iou
        data = {"WEIGHTS_PATH": WEIGHTS_PATH,
                        "imageFileName": id_,
                        "iou": iou,
                        "dice": dice}

        metrics = metrics.append(data, ignore_index=True)
    return metrics


def save_metrics(metrics, metrics_path):
    metrics_avg = metrics.groupby(['WEIGHTS_PATH'], as_index=False).mean()
    metrics_avg.to_csv(metrics_path, mode='a', index=False, header=False)


def check_all_weights(SAVE_PATH, X_test, Y_test, test_ids, metrics, METRICS_PATH, TRESHHOLD=0.5):
    inputs, outputs = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    model = create_model(inputs, outputs)
    model_ids = next(os.walk(SAVE_PATH))[2][:-1]
    for weights in model_ids:
        model.load_weights(filepath=SAVE_PATH + weights)
        metrics = model_predict(model, X_test, Y_test, test_ids, metrics, weights, TRESHHOLD)
    save_metrics(metrics, METRICS_PATH)


def check_weights(WEIGHTS_PATH, X_test, Y_test, test_ids, metrics, METRICS_PATH, TRESHHOLD=0.5):
    inputs, outputs = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    model = create_model(inputs, outputs)

    model.load_weights(filepath=WEIGHTS_PATH)
    metrics = model_predict(model, X_test, Y_test, test_ids, metrics, WEIGHTS_PATH, TRESHHOLD)
    save_metrics(metrics, METRICS_PATH)


def prepare_data(TEST_PATH, TEST_MASKS_PATH, TRESHHOLD):
    test_ids = next(os.walk(TEST_PATH))[2][:-1][:20]

    X_test = resizing_test_data(test_ids, TEST_PATH)
    Y_test = resizing_test_masks(test_ids, TEST_MASKS_PATH)
    Y_test = (Y_test > TRESHHOLD).astype(np.uint8)
    return test_ids, X_test, Y_test


def save_benchmark():
    test_ids, X_test, Y_test = prepare_data(TEST_PATH, TEST_MASKS_PATH, TRESHHOLD)
    metrics = pd.DataFrame(columns=['WEIGHTS_PATH', 'imageFileName', 'iou', 'dice'])
    check_all_weights(SAVE_PATH, X_test, Y_test, test_ids, metrics, METRICS_PATH, TRESHHOLD=0.5)
    # check_weights(WEIGHTS_PATH, X_test, Y_test, metrics, METRICS_PATH, TRESHHOLD=0.5)


def plot_iou(METRICS_PATH, title):
    metrics = pd.read_csv(METRICS_PATH)
    metrics["WEIGHTS_PATH"] = metrics["WEIGHTS_PATH"].str.slice(12,)
    plt.plot(metrics["WEIGHTS_PATH"], metrics["iou"])
    plt.title(title)
    plt.xlabel('WEIGHTS_PATH')
    plt.ylabel('iou')
    plt.xticks(rotation=20)
    plt.xticks(size=8)
    plt.yticks(size=10)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    TRESHHOLD = 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument("--TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tes/',
                        help='TEST PATH')
    parser.add_argument("--TEST_MASKS_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\2018_01_08_tes/',
                        help='TEST MASKS PATH')
    parser.add_argument("--SAVE_PATH", default=r'D:\Azure Repository\LNU_Course_work\data/', help='SAVE_PATH')
    parser.add_argument("--WEIGHTS_PATH", default=r"D:\Azure Repository\LNU_Course_work\data\u-net_model_epoch=13_valloss=0.0918.h5")
    parser.add_argument("--METRICS_PATH",
                        default=r"D:\Azure Repository\LNU_Course_work\U_Net_metrics.csv")

    args = parser.parse_args()
    TEST_PATH = args.TEST_PATH
    TEST_MASKS_PATH = args.TEST_MASKS_PATH
    SAVE_PATH = args.SAVE_PATH
    WEIGHTS_PATH = args.WEIGHTS_PATH
    METRICS_PATH = args.METRICS_PATH
    # save_benchmark()
    plot_iou(METRICS_PATH, "U-Net Model")

