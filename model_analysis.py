from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import pandas as pd
from metrics import *
from model_predict import *


def get_fp_tp(masks, masks_pred):
    thresholds = np.linspace(0, 1, 50)
    columns = ['threshold', 'false_positive_rate', 'true_positive_rate']
    inputs = pd.DataFrame(columns=columns)
    for i, threshold in enumerate(thresholds):
        mask_pred_t = (masks_pred > threshold).astype(np.uint8)
        tpr_avg = TPR_coef(masks.ravel(), mask_pred_t.ravel())
        fpr_avg = FPR_coef(masks.ravel(), mask_pred_t.ravel())
        inputs.loc[i, 'threshold'] = threshold
        inputs.loc[i, 'false_positive_rate'] = fpr_avg
        inputs.loc[i, 'true_positive_rate'] = tpr_avg
    return inputs


def get_precision_recall(masks, masks_pred):
    thresholds = np.linspace(0, 1, 100)
    columns = ['threshold', 'precision_rate', 'recall_rate']
    inputs = pd.DataFrame(columns=columns)
    for i, threshold in enumerate(thresholds):
        mask_pred_t = (masks_pred > threshold).astype(np.uint8)
        precision_avg = precision_coef(masks.ravel(), mask_pred_t.ravel())
        recall_avg = recall_coef(masks.ravel(), mask_pred_t.ravel())
        inputs.loc[i, 'threshold'] = threshold
        inputs.loc[i, 'precision_rate'] = precision_avg
        inputs.loc[i, 'recall_rate'] = recall_avg
    return inputs


def build_roc_curve(architectures, WEIGHTS_PATH, TEST_PATH, MASK_TEST_PATH):
    fprs, tprs, roc_aucs = [], [], []
    for architecture, weight in zip(architectures, WEIGHTS_PATH):
        if architecture == 'U-Net':
            model = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
        elif architecture == 'FCN':
            model = FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
        else:
            return
        _, X_test, Y_test, Y_pred = predict_images(model, TEST_PATH, MASK_TEST_PATH, weight, "no", 20)

        inputs = get_fp_tp(Y_test, Y_pred)
        fpr, tpr = inputs['false_positive_rate'], inputs['true_positive_rate']

        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)
    return fprs, tprs, roc_aucs


def build_precision_recall_curve(architectures, WEIGHTS_PATH, TEST_PATH, MASK_TEST_PATH):
    precisions, recols, pr_aucs = [], [], []
    for architecture, weight in zip(architectures, WEIGHTS_PATH):
        if architecture == 'U-Net':
            model = U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
        elif architecture == 'FCN':
            model = FCN(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
        else:
            return
        _, X_test, Y_test, Y_pred = predict_images(model, TEST_PATH, MASK_TEST_PATH, weight, "no", 20)
        inputs = get_precision_recall(Y_test, Y_pred)
        precision, recall = inputs['precision_rate'], inputs['recall_rate']
        pr_auc = auc(recall, precision)
        precisions.append(precision)
        recols.append(recall)
        pr_aucs.append(pr_auc)
    return precisions, recols, pr_aucs

