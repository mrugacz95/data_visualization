import unittest
import numpy as np

from evd import EVD_decomposition


class TestEVD(unittest.TestCase):

    def test_nonsymmetric_evd(self):
        arr = np.random.rand(5, 5)
        k, l, k_inv = EVD_decomposition(arr)
        assert np.allclose(np.diag(np.diag(l)), l)
        assert np.allclose(k @ l @ k_inv, arr)

    def test_symmetric_evd(self):
        arr = np.random.rand(5, 5)
        arr = arr + arr.T
        k, l, k_inv = EVD_decomposition(arr)
        assert np.allclose(np.diag(np.diag(l)), l)
        assert np.allclose(k @ l @ k_inv, arr)


if __name__ == '__main__':
    unittest.main()
