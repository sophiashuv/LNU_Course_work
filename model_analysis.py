from sklearn.metrics import roc_curve, auc
import pandas as pd
from matrices import *
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
