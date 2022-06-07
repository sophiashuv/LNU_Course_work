from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import pandas as pd
from metrics import *
from model_predict import *


def get_results(way, architecture, WEIGHT_PATH, TEST_PATH, MASK_TEST_PATH, PRED_PATH=None, FORMAT=None):
    if way == "1":
        if architecture == 'U-Net':
            model = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
        elif architecture == 'FCN-32':
            model = FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
        else:
            return
        test_ids, X_test, Y_test, Y_pred = predict_images(model, TEST_PATH, MASK_TEST_PATH, WEIGHT_PATH, "no", amount=-1)

    elif way == "2":
        _, Y_test, Y_pred = read_prediction_masks(MASK_TEST_PATH, PRED_PATH, "no", amount=-1, format=FORMAT)
    else:
        return
    return Y_test, Y_pred


def get_fp_tp(masks, masks_pred, inputs, architecture):
    thresholds = np.linspace(-0.9, 1.1, 50)

    for i, threshold in enumerate(thresholds):
        mask_pred_t = (masks_pred > threshold).astype(np.uint8)
        tpr_avg = TPR_coef(masks.ravel(), mask_pred_t.ravel())
        fpr_avg = FPR_coef(masks.ravel(), mask_pred_t.ravel())
        data = {"Model": architecture,
                "threshold": threshold,
                "false_positive_rate": fpr_avg,
                "true_positive_rate": tpr_avg,
                }
        inputs = inputs.append(data, ignore_index=True)
    return inputs


def get_precision_recall(masks, masks_pred, inputs, architecture):
    thresholds = np.linspace(-0.9, 1.1, 100)
    for i, threshold in enumerate(thresholds):
        mask_pred_t = (masks_pred > threshold).astype(np.uint8)
        precision_avg = precision_coef(masks.ravel(), mask_pred_t.ravel())
        recall_avg = recall_coef(masks.ravel(), mask_pred_t.ravel())

        data = {"Model": architecture,
                "threshold": threshold,
                "precision_rate": precision_avg,
                "recall_rate": recall_avg,
                }
        inputs = inputs.append(data, ignore_index=True)
    return inputs


def build_roc_curve(architectures, way, WEIGHTS_PATH, TEST_PATH, MASK_TEST_PATH, PRED_PATH=None, FORMAT=None):
    columns = ['Model', 'threshold', 'false_positive_rate', 'true_positive_rate']
    inputs = pd.DataFrame(columns=columns)
    for i, (architecture, weight) in enumerate(zip(architectures, WEIGHTS_PATH)):
        Y_test, Y_pred = get_results(way, architecture, weight, TEST_PATH, MASK_TEST_PATH, PRED_PATH[i], FORMAT)
        inputs = get_fp_tp(Y_test, Y_pred, inputs, architecture)
    return inputs


def build_precision_recall_curve(architectures, way, WEIGHTS_PATH, TEST_PATH, MASK_TEST_PATH, PRED_PATH=None, FORMAT=None):
    columns = ['Model', 'threshold', 'precision_rate', 'recall_rate']
    inputs = pd.DataFrame(columns=columns)
    for i, (architecture, weight) in enumerate(zip(architectures, WEIGHTS_PATH)):
        Y_test, Y_pred = get_results(way, architecture, weight, TEST_PATH, MASK_TEST_PATH, PRED_PATH[i], FORMAT)
        inputs = get_precision_recall(Y_test, Y_pred, inputs, architecture)
    return inputs
