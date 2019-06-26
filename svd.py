from sklearn.utils.extmath import randomized_svd
import numpy as np

from evd import EVD_decomposition

np.seterr(divide='ignore', invalid='ignore')


def sklearn_svd_implementation(channel, k):
    u, s, vt = randomized_svd(channel,
                              n_components=k,
                              n_iter=5,
                              random_state=None)
    return u, np.diag(s), vt


def numpy_svd_implementation(channel, k):
    u, s, vt = np.linalg.svd(channel)
    return u, np.diag(s), vt


def custom_svd_implementation(a, k):
    def to_square(arr):
        arr = arr.copy()
        new_size = min(arr.shape)
        return arr[:new_size, :new_size]

    def pseudo_inv(arr):
        arr = arr.copy()
        indices = np.nonzero(arr)
        arr[indices] = 1 / arr[indices]
        return arr

    m, n = a.shape
    if m > n:
        cnn = a.T @ a
        v, lv, vt = EVD_decomposition(cnn, True)

        # calculate T matrix
        t = np.zeros((m, n))
        diag = np.diag(lv)
        # remove values below zero from floating-point errors
        diag = diag.clip(min=0)
        t[:n, :n] = np.diag(np.sqrt(diag))

        # calculate U matrix
        u = np.zeros((m, m))
        u[:, :n] = a @ v @ pseudo_inv(to_square(t))
        # fill rest of columns with linear independent vectors
        u[:, n:] = np.random.rand(m, m - n)
        # make them orthogonal
        q, r = np.linalg.qr(u)  # changed to QR decomposition from numpy
        u = q @ r
    else:
        rmm = a @ a.T
        u, lu, ut = EVD_decomposition(rmm, True)

        # calculate T matrix
        t = np.zeros((m, n))
        diag = np.diag(lu)
        # remove values below zero from floating-point errors
        diag = diag.clip(min=0)
        t[:m, :m] = np.diag(np.sqrt(diag))

        # calculate V matrix
        v = np.zeros((n, n))
        v[:m, :] = pseudo_inv(to_square(t)) @ u.T @ a
        # fill rest of rows with linear independent vectors
        v[m:, :] = np.random.rand(n - m, n)
        # make them orthogonal
        q, r = np.linalg.qr(v.T)
        vt = (q @ r).T
    if k <= 0:
        return u @ t @ vt
    return u, t, vt
