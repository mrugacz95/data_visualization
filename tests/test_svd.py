import unittest

import numpy as np

from svd import custom_svd_implementation


class TestSVD(unittest.TestCase):

    def test_nonsymmetric_svd(self):
        arr = np.random.rand(5, 5)
        u, t, vt = custom_svd_implementation(arr, 5)
        assert np.allclose(np.diag(np.diag(t)), t)
        assert np.allclose(u @ t @ vt, arr)


if __name__ == '__main__':
    unittest.main()
