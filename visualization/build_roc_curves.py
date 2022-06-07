import sys
sys.path.insert(0, '..')

from plot_analysis import *


def plot_roc_curve(inputs):
    architectures = set(inputs["Model"])
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize="x-large")
    ax.set_ylabel('True Positive Rate', fontsize="x-large")
    ax.set_title('Receiver operating characteristic', fontsize="x-large")
    for architecture in architectures:
        color = np.random.randint(255, size=3) / 255
        fpr = inputs[inputs["Model"] == architecture]["false_positive_rate"]
        tpr = inputs[inputs["Model"] == architecture]["true_positive_rate"]
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label='ROC curve %s (area = %0.3f)' % (architecture, roc_auc), linewidth=3, color = color)
    ax.legend(loc="lower right", fontsize="x-large")
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.show()


def plot_precision_recall_curve(inputs):
    architectures = set(inputs["Model"])
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot([0, 0], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize="x-large")
    ax.set_ylabel('Precision', fontsize="x-large")
    ax.set_title('Precision-Recall Curves', fontsize="x-large")
    for architecture in architectures:
        color = np.random.randint(255, size=3) / 255
        precision = inputs[inputs["Model"] == architecture]["precision_rate"]
        recall = inputs[inputs["Model"] == architecture]["recall_rate"]
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label='Precision-Recall Curves %s (area = %0.3f)' % (architecture, pr_auc), linewidth=3, color=color)
    ax.legend(loc="lower right", fontsize="x-large")
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.show()


def plot_comparison(task):
    if task == '3':
        inputs = build_roc_curve(MODELS, WAY, WEIGHTS_PATH, TEST_PATH, MASK_TEST_PATH, PRED_PATH, FORMAT)
        plot_roc_curve(inputs)
    elif task == '4':
        inputs = build_precision_recall_curve(MODELS, WAY, WEIGHTS_PATH, TEST_PATH, MASK_TEST_PATH, PRED_PATH, FORMAT)
        plot_precision_recall_curve(inputs)


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--task",
                            default='3',
                            help='3 - plot_roc_curve 4 - plot_precision_recall_curve')
        parser.add_argument("--way",
                            default='1',
                            help='1 - predict images 2 - read predictions')
        parser.add_argument("--TEST_PATH",
                            default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\images\2018_01_08_tes/',
                            help='TEST PATH')
        parser.add_argument("--MASK_TEST_PATH",
                            default=r'C:\Users\sophi\OneDrive\Desktop\inherited_dataset\masks\2018_01_08_tes/',
                            help='TEST MASKS PATH')
        parser.add_argument('-n', '--MODELS', nargs='+', default=["U-Net", "FCN-32"])
        parser.add_argument('--PRED_PATH', nargs='+', default=[r"D:\Azure Repository\LNU_Course_work\rubbish\U-Net/", r"D:\Azure Repository\LNU_Course_work\rubbish\FCN-32/"])
        parser.add_argument('--WEIGHTS_PATH', nargs='+', default=[r"D:\Azure Repository\LNU_Course_work\U_Net_data\U-Net_model_epoch=10_valloss=0.0812.h5", r"D:\Azure Repository\LNU_Course_work\FCN32_data\FCN-32_model_epoch=7_valloss=0.0885.h5"])
        parser.add_argument("--FORMAT",
                            default='.jpg',
                            help='Possible: .npy, .png')

        args = parser.parse_args()
        task = args.task

        TEST_PATH = args.TEST_PATH
        WAY = args.way
        MASK_TEST_PATH = args.MASK_TEST_PATH
        MODELS = args.MODELS
        PRED_PATH = args.PRED_PATH
        WEIGHTS_PATH = args.WEIGHTS_PATH
        FORMAT = args.FORMAT

        plot_comparison(task)
