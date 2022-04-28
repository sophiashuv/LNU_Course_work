
import unittest
from unittest import TestCase

from metrics import *
from show_prediction import *


class TestKmeam(TestCase):
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

    def test_im1(self):
        tpr_res = TPR_coef(self.mask1, self.pred_mask1)
        fpr_res = FPR_coef(self.mask1, self.pred_mask1)
        np.testing.assert_almost_equal(tpr_res, 15.0/25)
        np.testing.assert_almost_equal(fpr_res, 0)

    def test_im2(self):
        tpr_res = TPR_coef(self.mask2, self.pred_mask2)
        fpr_res = FPR_coef(self.mask2, self.pred_mask2)
        np.testing.assert_almost_equal(tpr_res, 1)
        np.testing.assert_almost_equal(fpr_res, 10/85)

    def test_im3(self):
        tpr_res = TPR_coef(self.mask3, self.pred_mask3)
        fpr_res = FPR_coef(self.mask3, self.pred_mask3)
        np.testing.assert_almost_equal(tpr_res, 15.0/25)
        np.testing.assert_almost_equal(fpr_res, 10.0/75)

    def test_im4(self):
        tpr_res = TPR_coef(self.mask4, self.pred_mask4)
        fpr_res = FPR_coef(self.mask4, self.pred_mask4)
        np.testing.assert_almost_equal(tpr_res, 9.0/25)
        np.testing.assert_almost_equal(fpr_res, 16.0/75)


if __name__ == '__main__':
    unittest.main()
