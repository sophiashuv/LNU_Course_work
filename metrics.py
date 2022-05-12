import math
import numpy as np


def get_confusion_matrix(y_true, y_pred):
    difference = y_pred - y_true
    difference2 = y_true - y_pred
    sum = y_true + y_pred

    TP = sum[sum == 2].shape[0]
    FN = difference[difference2 == 1].shape[0]
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
        return None
    return TP/(FP + TP)


def recall_coef(y_true, y_pred):
    TN, FP, FN, TP = get_confusion_matrix(y_true, y_pred)
    if FN + TP == 0:
        return None
    return TP/(FN + TP)


def TPR_coef(y_true, y_pred):
    TN, FP, FN, TP = get_confusion_matrix(y_true, y_pred)
    if FN + TP == 0:
        return None
    return TP/(FN + TP)


def FPR_coef(y_true, y_pred):
    TN, FP, FN, TP = get_confusion_matrix(y_true, y_pred)
    if FP + TN == 0:
        return None
    return FP/(FP + TN)


def mcc_coef(y_true, y_pred):
    TN, FP, FN, TP = get_confusion_matrix(y_true, y_pred)
    if (TP + FP == 0) or (TP + FN == 0) or (TN + FP == 0) or (TN + FN == 0):
        return 0
    return (TP * TN - FP * FN)/(math.sqrt((TP + FP)) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN))


def F1_coef(y_true, y_pred):
    TN, FP, FN, TP = get_confusion_matrix(y_true, y_pred)
    if TP + (FP + FN)/2:
        return 0
    return TP / (TP + (FP + FN)/2)
