import re

from models_comparison import *
from model_analysis import *


def plot_metrics(models, x, ys, y_value):
    plt.figure(figsize=(18, 9))
    plt.title("Models Comparison based on concrete images")
    plt.xlabel('Image', fontweight='bold')
    plt.ylabel(y_value, fontweight='bold')
    plt.xticks(size=7)
    plt.yticks(size=10)
    plt.grid(which='major', color='#CCCCCC', linestyle='--')

    for y, model in zip(ys, models):
        color = np.random.randint(255, size=3) / 255
        xlabels_new = [re.sub("(.{10})", "\\1\n", label, 0, re.DOTALL) for label in x]
        # plt.plot(xlabels_new, y, 'o', markersize=20, color=color, label=model)
        plt.plot(xlabels_new, y, '-ok', color=color, label=model)
        plt.legend(loc="upper left", title="Models", fontsize=12, title_fontsize=15)
    if y_value == 'mcc':
        plt.ylim(-1.1, 1.7)
    else:
        plt.ylim(-0.1, 1.7)
    plt.show()


def plot_average(metrics):
    colors = ["#85C1E9", "#F7DC6F"]
    metrics.plot(y=["U-Net", "FCN"], use_index=True, color=colors, kind="bar", zorder=3, figsize=(18, 9))
    plt.title("Models Comparison")
    plt.xticks(size=12)
    plt.yticks(size=15)
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.show()


def plot_roc_curve(architectures, fprs, tprs, roc_aucs):
    fig, ax = plt.subplots(1, 1)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    for architecture, fpr, tpr, roc_auc in zip(architectures, fprs, tprs, roc_aucs):
        ax.plot(fpr, tpr, label='ROC curve %s (area = %0.2f)' % (architecture, roc_auc))
    ax.legend(loc="lower right")
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.show()


def plot_precision_recall_curve(architectures, precisions, recols, pr_aucs):
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_title('Precision-Recall Curves')
    for architecture, recall, precision, pr_auc in zip(architectures, recols, precisions,  pr_aucs):
        ax.plot(precision, recall, label='Precision-Recall Curves %s (area = %0.2f)' % (architecture, pr_auc))
    ax.legend(loc="lower right")
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.show()


def plot_comparison(task):
    if task == '1':
        models, x, ys = get_image_metrics(METRICS_PATH, metric, 20, THRESHOLD)
        plot_metrics(models, x, ys, metric)
    elif task == '2':
        metrics = get_model_avg(METRICS_PATH, THRESHOLD)
        plot_average(metrics)
    elif task == '3':
        architectures = ["U-Net", "FCN"]
        fprs, tprs, roc_aucs = build_roc_curve(architectures, way, [WEIGHTS_PATH_UNET, WEIGHTS_PATH_FCN], TEST_PATH, MASK_TEST_PATH, [PRED_PATH_UNet, PRED_PATH_FCN], FORMAT)
        plot_roc_curve(architectures, fprs, tprs, roc_aucs)
    elif task == '4':
        architectures = ["U-Net", "FCN"]
        precisions, recols, pr_aucs = build_precision_recall_curve(architectures, way, [WEIGHTS_PATH_UNET, WEIGHTS_PATH_FCN], TEST_PATH, MASK_TEST_PATH, [PRED_PATH_UNet, PRED_PATH_FCN], FORMAT)
        plot_precision_recall_curve(architectures, precisions, recols, pr_aucs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        default='3',
                        help='1 - plot_metrics, 2 - plot_average 3 - plot_roc_curve 4 - plot_precision_recall_curve')
    parser.add_argument("--way",
                        default='2',
                        help='1 - predict, 2 - read predictions')
    parser.add_argument('--THRESHOLD',
                        default=0.3)
    parser.add_argument("--TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tes/',
                        help='TEST PATH')
    parser.add_argument("--MASK_TEST_PATH",
                        default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\2018_01_08_tes/',
                        help='TEST MASKS PATH')
    parser.add_argument("--PRED_PATH_UNet",
                        default=r'D:\Azure Repository\LNU_Course_work\rubbish\U-Net/',
                        help='PRED_PATH')
    parser.add_argument("--PRED_PATH_FCN",
                        default=r'D:\Azure Repository\LNU_Course_work\rubbish\FCN/',
                        help='PRED_PATH')
    parser.add_argument("--FORMAT",
                        default='.npy',
                        help='Possible: .npu, .png')
    parser.add_argument("--WEIGHTS_PATH_UNET",
                        default=r"D:\Azure Repository\LNU_Course_work\U_Net_data\u-net_model_epoch=10_valloss=0.1094.h5")
    parser.add_argument("--WEIGHTS_PATH_FCN",
                        default=r"D:\Azure Repository\LNU_Course_work\FCN_data\FCN_model_epoch=5_valloss=0.1269.h5")
    parser.add_argument("--metric",
                        default='iou',
                        help='"iou", "accuracy", "precision", "recall", "mcc"')
    parser.add_argument("--METRICS_PATH",
                        default=r"D:\Azure Repository\LNU_Course_work\metrics\metrics.csv")

    args = parser.parse_args()
    task = args.task
    way = args.way
    THRESHOLD = args.THRESHOLD
    metric = args.metric
    METRICS_PATH = args.METRICS_PATH
    TEST_PATH = args.TEST_PATH
    MASK_TEST_PATH = args.MASK_TEST_PATH
    PRED_PATH_UNet = args.PRED_PATH_UNet
    PRED_PATH_FCN = args.PRED_PATH_FCN
    FORMAT = args.FORMAT
    WEIGHTS_PATH_UNET = args.WEIGHTS_PATH_UNET
    WEIGHTS_PATH_FCN = args.WEIGHTS_PATH_FCN

    plot_comparison(task)
