import math

import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from train_model import *
from data_preparation import *


def get_confusion_matrix(y_true, y_pred):
    TP = np.sum(y_pred * y_true)
    difference = y_pred - y_true
    sum = y_true + y_pred
    FN = difference[difference == -1].shape[0]
    FP = difference[difference == 1].shape[0]
    TN = sum[sum == 0].shape[0]
    return TN, FP, FN, TP


def jacard_coef(y_true, y_pred):
    intersection = np.sum(y_pred * y_true)
    union = np.sum(y_pred) + np.sum(y_true) - intersection
    if union == 0:
        return 0
    return intersection / union


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef(y_true, y_pred):
    return 1 - jacard_coef(y_true, y_pred)


def accuracy_coef(y_true, y_pred):
    TN, FP, FN, TP = get_confusion_matrix(y_true, y_pred)
    return (TP + TN)/(TN + FP + FN + TP)


def precision_coef(y_true, y_pred):
    TN, FP, FN, TP = get_confusion_matrix(y_true, y_pred)
    if FP + TP == 0:
        return 0
    return TP/(FP + TP)


def recall_coef(y_true, y_pred):
    TN, FP, FN, TP = get_confusion_matrix(y_true, y_pred)
    if FN + TP == 0:
        return 0
    return TP/(FN + TP)


def mcc_coef(y_true, y_pred):
    TN, FP, FN, TP = get_confusion_matrix(y_true, y_pred)
    if (TP + FP == 0) or (TP + FN == 0) or (TN + FP == 0) or (TN + FN == 0):
        return 0
    return (TP * TN - FP * FN)/(math.sqrt((TP + FP)) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN))


def model_predict(model, X_test, WEIGHTS_PATH, treshhold=0.5):
    model.load_weights(filepath=SAVE_PATH + WEIGHTS_PATH)
    preds_test = model.predict(X_test, verbose=1)
    preds_test_t = (preds_test > treshhold).astype(np.uint8)
    preds_test_t = np.array(
        [np.reshape(
            cv2.resize(mask, (BIG_IMG_WIDTH, BIG_IMG_HEIGHT)),
            (BIG_IMG_HEIGHT, BIG_IMG_WIDTH, 1))
            for mask in preds_test_t])
    return preds_test_t


def get_metrics_df(preds_test_t, Y_test, test_ids, metrics, WEIGHTS_PATH):
    for predicted_mask, groundtruth_mask, id_ in zip(Y_test, preds_test_t, test_ids):
        iou = jacard_coef(groundtruth_mask, predicted_mask)
        dice = dice_coef(groundtruth_mask, predicted_mask)
        accuracy = accuracy_coef(groundtruth_mask, predicted_mask)
        precision = precision_coef(groundtruth_mask, predicted_mask)
        recall = recall_coef(groundtruth_mask, predicted_mask)
        mcc = mcc_coef(groundtruth_mask, predicted_mask)
        data = {"WEIGHTS_PATH": WEIGHTS_PATH,
                        "imageFileName": id_,
                        "iou": iou,
                        "dice": dice,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "mcc": mcc
                }
        metrics = metrics.append(data, ignore_index=True)
    return metrics


def save_metrics_to_file(metrics, metrics_path):
    metrics.to_csv(metrics_path, mode='a', index=False, header=False)


def check_all_weights(model, SAVE_PATH, X_test, Y_test, test_ids, metrics, METRICS_PATH):
    model_ids = next(os.walk(SAVE_PATH))[2]
    for weights in model_ids:
        preds_test_t = model_predict(model, X_test, weights)
        metrics = get_metrics_df(preds_test_t, Y_test, test_ids, metrics, weights)
    save_metrics_to_file(metrics, METRICS_PATH)


def check_weights(model, WEIGHTS_PATH, X_test, Y_test, test_ids, metrics, METRICS_PATH):
    preds_test_t = model_predict(model, X_test, WEIGHTS_PATH)
    metrics = get_metrics_df(preds_test_t, Y_test, test_ids, metrics, WEIGHTS_PATH)
    save_metrics_to_file(metrics, METRICS_PATH)


def prepare_data(TEST_PATH, TEST_MASKS_PATH, TRESHHOLD):
    test_ids = next(os.walk(TEST_PATH))[2][:-1][:20]
    X_test = resizing_test_data(test_ids, TEST_PATH)
    Y_test = resizing_test_masks(test_ids, TEST_MASKS_PATH)
    Y_test = (Y_test > TRESHHOLD).astype(np.uint8)
    return test_ids, X_test, Y_test


def save_benchmark(model):
    test_ids, X_test, Y_test = prepare_data(TEST_PATH, TEST_MASKS_PATH, TRESHHOLD)
    metrics = pd.DataFrame(columns=['WEIGHTS_PATH', 'imageFileName', 'iou', 'dice', 'accuracy', 'precision', 'recall'])
    check_all_weights(model, SAVE_PATH, X_test, Y_test, test_ids, metrics, METRICS_PATH)


def plot_metric(METRICS_PATH, title, y_value):
    metrics = pd.read_csv(METRICS_PATH)
    metrics["WEIGHTS_PATH"] = metrics["WEIGHTS_PATH"].str.slice(len(title) + 7,)
    weights = metrics["WEIGHTS_PATH"]
    plt.figure(figsize=(18, 9))
    plt.title(title + " Model")
    plt.xlabel('WEIGHTS_PATH', fontweight='bold')
    plt.ylabel(y_value, fontweight='bold')
    plt.xticks(size=7)
    plt.yticks(size=10)
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    weights = set(weights)
    for weight in weights:
        weight_metrics = metrics[metrics["WEIGHTS_PATH"] == weight]
        color = np.random.randint(255, size=3)/255
        xlabels_new = [re.sub("(.{10})", "\\1\n", label, 0, re.DOTALL) for label in weight_metrics["imageFileName"]]
        plt.plot(xlabels_new, weight_metrics[y_value], color=color, label=weight)
        plt.legend(loc="upper left")
        if y_value == 'mcc':
            plt.ylim(-1.1, 1.7)
        else:
            plt.ylim(-0.1, 1.7)
    plt.show()


def plot_metric_avg(METRICS_PATH, title, y_value):
    metrics = pd.read_csv(METRICS_PATH)
    metrics = metrics.groupby(['WEIGHTS_PATH'], as_index=False).mean()
    metrics["WEIGHTS_PATH"] = metrics["WEIGHTS_PATH"].str.slice(len(title) + 7,)
    plt.figure(figsize=(18, 9))
    xlabels_new = [re.sub("(.{10})", "\\1\n", label, 0, re.DOTALL) for label in metrics["WEIGHTS_PATH"]]
    plt.bar(xlabels_new, metrics[y_value], width=0.3, color="#f7dc57", zorder=3)
    plt.title(title + " Model")
    plt.xlabel('WEIGHTS_PATH', fontweight='bold')
    plt.ylabel(y_value, fontweight='bold')
    plt.xticks(size=10)
    plt.yticks(size=10)
    if y_value == 'mcc':
        plt.ylim(-1.1, 1.7)
    else:
        plt.ylim(-0.1, 1.7)
    plt.grid(zorder=0, which='major', color='#CCCCCC', linestyle='--')
    plt.show()


def show_segmentation(model, img_id, WEIGHTS_PATH, TEST_PATH, TEST_MASKS_PATH):
    img = cv2.imread(TEST_PATH + img_id)[:, :, :IMG_CHANNELS]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    mask = cv2.imread(TEST_MASKS_PATH + img_id[:-3] + "png", 0)/255
    preds_test_t = model_predict(model, np.array([img_small]), WEIGHTS_PATH, treshhold=0.5)
    mask_pred = np.squeeze(preds_test_t)
    mask_pred = cv2.resize(mask_pred, (mask.shape[1], mask.shape[0]))

    sum = mask_pred + mask
    differ = mask_pred - mask
    color = np.zeros((img.shape), dtype=np.uint8)
    color[differ == 1] = np.array([219, 62, 50])
    color[differ == -1] = np.array([83, 161, 67])
    color[sum == 2] = np.array([31, 117, 204])

    img[sum != 0] = cv2.addWeighted(img[sum != 0], 0.6, color[sum != 0], 0.4, 0)

    plt.figure(figsize=(18, 9))
    plt.imshow(img)
    patches = [mpatches.Patch(color="#db3e32", label="FALSE POSITIVE"),
               mpatches.Patch(color="#53a143", label="FALSE NEGATIVE"),
               mpatches.Patch(color="#1f75cc", label="TRUE POSITIVE")]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def analyse(task, architecture, metrics):
    if architecture == 'U-Net':
        model = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    elif architecture == 'FCN':
        model = FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    else:
        return
    if task == '1':
        save_benchmark(model)
    elif task == '2':
        plot_metric(METRICS_PATH, architecture, metrics)
    elif task == '3':
        plot_metric_avg(METRICS_PATH, architecture, metrics)
    elif task == '4':
        show_segmentation(model, IMG_PATH, WEIGHTS_PATH, TEST_PATH, TEST_MASKS_PATH)


if __name__ == '__main__':
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    TRESHHOLD = 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        default='4',
                        help='1 - save_benchmark, 2 - plot_metric, 3 - plot_metric_avg, 4 - show_segmentation')
    parser.add_argument('--architecture',
                        default='U-Net',
                        help='Possible: U-Net, FCN')
    parser.add_argument("--TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tes/',
                        help='TEST PATH')
    parser.add_argument("--TEST_MASKS_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\2018_01_08_tes/',
                        help='TEST MASKS PATH')
    parser.add_argument("--SAVE_PATH", default=r'D:\Azure Repository\LNU_Course_work\U_Net_data/', help='SAVE_PATH')
    parser.add_argument("--WEIGHTS_PATH", default=r"u-net_model_epoch=10_valloss=0.1094.h5")
    parser.add_argument("--METRICS_PATH",
                        default=r"D:\Azure Repository\LNU_Course_work\metrics\U_Net_metrics.csv")
    parser.add_argument("--IMG_PATH",
                        default=r"5F781B80-71AA-4BE3-ABD7-0198659685C7.jpg")
    parser.add_argument("--metrics",
                        default=r"accuracy",
                        help='possible: iou, dice, accuracy, precision, recall, mcc')

    args = parser.parse_args()
    task = args.task
    architecture = args.architecture
    TEST_PATH = args.TEST_PATH
    TEST_MASKS_PATH = args.TEST_MASKS_PATH
    SAVE_PATH = args.SAVE_PATH
    WEIGHTS_PATH = args.WEIGHTS_PATH
    METRICS_PATH = args.METRICS_PATH
    metrics = args.metrics
    IMG_PATH = args.IMG_PATH
    analyse(task, architecture, metrics)

