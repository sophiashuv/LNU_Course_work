import re
import pandas as pd

from model_predict import *


def get_image_matrices(MATRICES_PATH, y_value, amount, THRESHOLD=0.3):
    matrices = pd.read_csv(MATRICES_PATH)
    matrices = matrices[matrices['THRESHOLD'] == THRESHOLD]
    if len(matrices) == 0:
        print("No data with this treshhold")
        return

    models = set(matrices["MODEL"])

    p = matrices.groupby(["imageFileName"], as_index=False).mean()
    p = p.sort_values([y_value])
    images = p["imageFileName"]

    y = []
    for model in models:
        weight_matrices = matrices[matrices["MODEL"] == model]
        weight_matrices = weight_matrices.set_index('imageFileName')
        weight_matrices = weight_matrices.reindex(index=p['imageFileName'])
        weight_matrices = weight_matrices.reset_index()
        y.append(weight_matrices[y_value])
    y = np.array(y)
    return models, images[:amount], y[:, :amount]


def get_model_avg(METRICS_PATH, THRESHOLD=0.3):
    matrices = pd.read_csv(METRICS_PATH)
    matrices = matrices[matrices['THRESHOLD'] == THRESHOLD]
    if len(matrices) == 0:
        print("No data with this threshold")
        return
    matrices = matrices.groupby(['MODEL'], as_index=False).mean()
    return matrices
