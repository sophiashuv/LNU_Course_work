import re
import sys
sys.path.insert(0, '..')

from models_comparison import *
from model_analysis import *


def plot_metrics(models, x, ys, y_value):
    plt.figure(figsize=(18, 5))
    plt.title(y_value + "-metric models Comparison based on concrete images")
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
    metrics.plot(y=["U-Net", "FCN-32"], use_index=True, color=colors, kind="bar", zorder=3, figsize=(18, 9))
    plt.title("Models Comparison")
    plt.xticks(size=12)
    plt.yticks(size=15)
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.show()


def plot_comparison(task):
    if task == '1':
        models, x, ys = get_image_metrics(METRICS_PATH, metric, 20, THRESHOLD)
        plot_metrics(models, x, ys, metric)
    elif task == '2':
        metrics = get_model_avg(METRICS_PATH, THRESHOLD)
        plot_average(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        default='1',
                        help='1 - plot_metrics, 2 - plot_average')
    parser.add_argument('--THRESHOLD',
                        default=0.3)
    parser.add_argument("--metric",
                        default='f1',
                        help='"iou", "accuracy", "precision", "recall", "mcc", "f1')
    parser.add_argument("--METRICS_PATH",
                        default=r"D:\Azure Repository\LNU_Course_work\metrics\metrics.csv")

    args = parser.parse_args()
    task = args.task
    THRESHOLD = args.THRESHOLD
    metric = args.metric
    METRICS_PATH = args.METRICS_PATH

    plot_comparison(task)
