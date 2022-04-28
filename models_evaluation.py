import pandas as pd

from metrics import *
from model_predict import *


def get_metrics_df(test_ids, Y_test, Y_pred, metrics, architecture, THRESHOLD):
    for predicted_mask, groundtruth_mask, id_ in zip(Y_test, Y_pred, test_ids):
        iou = jacard_coef(groundtruth_mask, predicted_mask)
        accuracy = accuracy_coef(groundtruth_mask, predicted_mask)
        precision = precision_coef(groundtruth_mask, predicted_mask)
        recall = recall_coef(groundtruth_mask, predicted_mask)
        mcc = mcc_coef(groundtruth_mask, predicted_mask)
        f1 = F1_coef(groundtruth_mask, predicted_mask)
        data = {"MODEL": architecture,
                "imageFileName": id_,
                "THRESHOLD": THRESHOLD,
                "iou": iou,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "mcc": mcc,
                'f1': f1
                }
        metrics = metrics.append(data, ignore_index=True)
    return metrics


def save_metrics_to_file(metrics, metrics_path):
    metrics.to_csv(metrics_path, mode='a', index=False, header=False)


def save_benchmark(architecture):
    if architecture == 'U-Net':
        model = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    elif architecture == 'FCN':
        model = FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    else:
        return

    test_ids, X_test, Y_test, Y_pred = predict_images(model, TEST_PATH, MASK_TEST_PATH, WEIGHTS_PATH, THRESHOLD)
    metrics = pd.DataFrame(columns=['MODEL', 'imageFileName', "THRESHOLD", 'iou', 'accuracy', 'precision', 'recall', 'f1'])
    metrics = get_metrics_df(test_ids, Y_test, Y_pred, metrics, architecture, THRESHOLD)
    save_metrics_to_file(metrics, METRICS_PATH)


if __name__ == '__main__':
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture',
                        default='U-Net',
                        help='Possible: U-Net, FCN')
    parser.add_argument('--THRESHOLD',
                        default=0.3)
    parser.add_argument("--TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tes/',
                        help='TEST PATH')
    parser.add_argument("--MASK_TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\2018_01_08_tes/',
                        help='TEST MASKS PATH')
    parser.add_argument("--WEIGHTS_PATH",
                        default=r"D:\Azure Repository\LNU_Course_work\U_Net_data\u-net_model_epoch=10_valloss=0.1094.h5")
    parser.add_argument("--METRICS_PATH",
                        default=r"D:\Azure Repository\LNU_Course_work\metrics\metrics.csv")
    args = parser.parse_args()
    architecture = args.architecture
    THRESHOLD = args.THRESHOLD
    TEST_PATH = args.TEST_PATH
    MASK_TEST_PATH = args.MASK_TEST_PATH
    WEIGHTS_PATH = args.WEIGHTS_PATH
    METRICS_PATH = args.METRICS_PATH

    save_benchmark(architecture)
