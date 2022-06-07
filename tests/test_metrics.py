
import unittest
from unittest import TestCase

from metrics import *


class TestMetrics(TestCase):
    def setUp(self):
        self.mask1 = np.full((10, 10), 0, dtype=np.uint8)
        self.mask1[:5, :5] = 1
        self.pred_mask1 = np.full((10, 10), 0, dtype=np.uint8)
        self.pred_mask1[:5, 2:5] = 1

        self.mask2 = np.full((10, 10), 0, dtype=np.uint8)
        self.mask2[:5, 2:5] = 1
        self.pred_mask2 = np.full((10, 10), 0, dtype=np.uint8)
        self.pred_mask2[:5, :5] = 1

        self.mask3 = np.full((10, 10), 0, dtype=np.uint8)
        self.mask3[:5, 2:7] = 1
        self.pred_mask3 = np.full((10, 10), 0, dtype=np.uint8)
        self.pred_mask3[:5, :5] = 1

        self.mask4 = np.full((10, 10), 0, dtype=np.uint8)
        self.mask4[2:7, 2:7] = 1
        self.pred_mask4 = np.full((10, 10), 0, dtype=np.uint8)
        self.pred_mask4[:5, :5] = 1

        self.mask5 = np.full((10, 10), 0, dtype=np.uint8)
        self.mask5[:5, :5] = 1
        self.pred_mask5 = np.full((10, 10), 0, dtype=np.uint8)
        self.pred_mask5[2:7, 2:7] = 1

    def test_im1_tpr(self):
        tpr_res = TPR_coef(self.mask1, self.pred_mask1)
        np.testing.assert_almost_equal(tpr_res, 15.0/25)

    def test_im2_tpr(self):
        tpr_res = TPR_coef(self.mask2, self.pred_mask2)
        np.testing.assert_almost_equal(tpr_res, 1)

    def test_im3_tpr(self):
        tpr_res = TPR_coef(self.mask3, self.pred_mask3)
        np.testing.assert_almost_equal(tpr_res, 15.0/25)

    def test_im4_tpr(self):
        tpr_res = TPR_coef(self.mask4, self.pred_mask4)
        np.testing.assert_almost_equal(tpr_res, 9.0/25)

    def test_im5_tpr(self):
        tpr_res = TPR_coef(self.mask5, self.pred_mask5)
        np.testing.assert_almost_equal(tpr_res, 9.0/25)

    def test_im1_fpr(self):
        fpr_res = FPR_coef(self.mask1, self.pred_mask1)
        np.testing.assert_almost_equal(fpr_res, 0)

    def test_im2_fpr(self):
        fpr_res = FPR_coef(self.mask2, self.pred_mask2)
        np.testing.assert_almost_equal(fpr_res, 10/85)

    def test_im3_fpr(self):
        fpr_res = FPR_coef(self.mask3, self.pred_mask3)
        np.testing.assert_almost_equal(fpr_res, 10.0/75)

    def test_im4_fpr(self):
        fpr_res = FPR_coef(self.mask4, self.pred_mask4)
        np.testing.assert_almost_equal(fpr_res, 16.0/75)

    def test_im5_fpr(self):
        fpr_res = FPR_coef(self.mask5, self.pred_mask5)
        np.testing.assert_almost_equal(fpr_res, 16.0/75)

    def test_im1_precision(self):
        precision_res = precision_coef(self.mask1, self.pred_mask1)
        np.testing.assert_almost_equal(precision_res, 1)

    def test_im2_precision(self):
        precision_res = precision_coef(self.mask2, self.pred_mask2)
        np.testing.assert_almost_equal(precision_res, 0.6)

    def test_im3_precision(self):
        precision_res = precision_coef(self.mask3, self.pred_mask3)
        np.testing.assert_almost_equal(precision_res, 0.6)

    def test_im4_precision(self):
        precision_res = precision_coef(self.mask4, self.pred_mask4)
        np.testing.assert_almost_equal(precision_res, 0.36)

    def test_im5_precision(self):
        precision_res = precision_coef(self.mask5, self.pred_mask5)
        np.testing.assert_almost_equal(precision_res, 0.36)

    def test_im1_jacard(self):
        jacard_res = jacard_coef(self.mask1, self.pred_mask1)
        np.testing.assert_almost_equal(jacard_res, 0.6)

    def test_im2_jacard(self):
        jacard_res = jacard_coef(self.mask2, self.pred_mask2)
        np.testing.assert_almost_equal(jacard_res, 0.6)

    def test_im3_jacard(self):
        jacard_res = jacard_coef(self.mask3, self.pred_mask3)
        np.testing.assert_almost_equal(jacard_res, 0.42857142857142855)

    def test_im4_jacard(self):
        jacard_res = jacard_coef(self.mask4, self.pred_mask4)
        np.testing.assert_almost_equal(jacard_res, 0.21951219512195122)

    def test_im5_jacard(self):
        jacard_res = jacard_coef(self.mask5, self.pred_mask5)
        np.testing.assert_almost_equal(jacard_res, 0.21951219512195122)

    def test_im1_accuracy(self):
        accuracy_res = accuracy_coef(self.mask1, self.pred_mask1)
        np.testing.assert_almost_equal(accuracy_res, 0.9)

    def test_im2_accuracy(self):
        accuracy_res = accuracy_coef(self.mask2, self.pred_mask2)
        np.testing.assert_almost_equal(accuracy_res, 0.9)

    def test_im3_accuracy(self):
        accuracy_res = accuracy_coef(self.mask3, self.pred_mask3)
        np.testing.assert_almost_equal(accuracy_res, 0.8)

    def test_im4_accuracy(self):
        accuracy_res = accuracy_coef(self.mask4, self.pred_mask4)
        np.testing.assert_almost_equal(accuracy_res, 0.68)

    def test_im5_accuracy(self):
        accuracy_res = accuracy_coef(self.mask5, self.pred_mask5)
        np.testing.assert_almost_equal(accuracy_res, 0.68)

    def test_im1_F1(self):
        F1_res = F1_coef(self.mask1, self.pred_mask1)
        np.testing.assert_almost_equal(F1_res, 0.75)

    def test_im2_F1(self):
        F1_res = F1_coef(self.mask2, self.pred_mask2)
        np.testing.assert_almost_equal(F1_res, 0.75)

    def test_im3_F1(self):
        F1_res = F1_coef(self.mask3, self.pred_mask3)
        np.testing.assert_almost_equal(F1_res, 0.6)

    def test_im4_F1(self):
        F1_res = F1_coef(self.mask4, self.pred_mask4)
        np.testing.assert_almost_equal(F1_res, 0.36)

    def test_im5_F1(self):
        F1_res = F1_coef(self.mask5, self.pred_mask5)
        np.testing.assert_almost_equal(F1_res, 0.36)

    def test_im1_mcc(self):
        mcc_res = mcc_coef(self.mask1, self.pred_mask1)
        np.testing.assert_almost_equal(mcc_res,  0.7276068751089989)

    def test_im2_mcc(self):
        mcc_res = mcc_coef(self.mask2, self.pred_mask2)
        np.testing.assert_almost_equal(mcc_res,  0.7276068751089989)

    def test_im3_mcc(self):
        mcc_res = mcc_coef(self.mask3, self.pred_mask3)
        np.testing.assert_almost_equal(mcc_res, 0.4666666666666666)

    def test_im4_mcc(self):
        mcc_res = mcc_coef(self.mask4, self.pred_mask4)
        np.testing.assert_almost_equal(mcc_res, 0.14666666666666664)

    def test_im5_mcc(self):
        mcc_res = mcc_coef(self.mask5, self.pred_mask5)
        np.testing.assert_almost_equal(mcc_res, 0.14666666666666664)


if __name__ == '__main__':
    unittest.main()
