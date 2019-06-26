import numpy as np


def EVD_decomposition(arr, symmetric_matrix=False):
    if symmetric_matrix:
        w, v = np.linalg.eigh(arr)
    else:
        w, v = np.linalg.eig(arr)
    indices = np.argsort(w)[::-1]
    w = w[indices]
    v = v[:, indices]
    k = v
    k_inv = np.linalg.inv(v)
    l = np.diag(w)
    return k, l, k_inv
