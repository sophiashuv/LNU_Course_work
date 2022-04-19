import re
import pandas as pd

from model_predict import *


def get_image_metrics(METRICS_PATH, y_value, amount, THRESHOLD=0.3):
    metrics = pd.read_csv(METRICS_PATH)
    metrics = metrics[metrics['THRESHOLD'] == THRESHOLD]
    if len(metrics) == 0:
        print("No data with this treshhold")
        return

    models = set(metrics["MODEL"])

    p = metrics.groupby(["imageFileName"], as_index=False).mean()
    p = p.sort_values([y_value])
    images = p["imageFileName"]

    y = []
    for model in models:
        weight_metrics = metrics[metrics["MODEL"] == model]
        weight_metrics = weight_metrics.set_index('imageFileName')
        weight_metrics = weight_metrics.reindex(index=p['imageFileName'])
        weight_metrics = weight_metrics.reset_index()
        y.append(weight_metrics[y_value])
    y = np.array(y)
    return models, images[:amount], y[:, :amount]


def get_model_avg(METRICS_PATH, THRESHOLD=0.3):
    metrics = pd.read_csv(METRICS_PATH)
    metrics = metrics[metrics['THRESHOLD'] == THRESHOLD]
    if len(metrics) == 0:
        print("No data with this threshold")
        return
    metrics = metrics.groupby(['MODEL'], as_index=False).mean()
    metrics = metrics.T
    metrics.rename(columns=metrics.iloc[0], inplace=True)
    metrics.drop(metrics.index[0], inplace=True)
    metrics.drop(metrics.index[0], inplace=True)
    return metrics
