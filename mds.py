from scipy import spatial
from sklearn import manifold, preprocessing
import numpy as np


def centering_matrix(n):
    return np.identity(n) - 1 / n * np.ones(n)


def mds_custom(data):
    n = 2
    examp_num, atrbib_num = data.shape
    # standardization
    data = preprocessing.scale(data)
    # distances
    D = spatial.distance.cdist(data, data)
    # B matrix
    B = - 1 / 2 * centering_matrix(examp_num) @ (D ** 2) @ centering_matrix(examp_num)
    # eigenval, eigenvec
    eigenval, eigenvec = np.linalg.eig(B)
    # sorting
    indices = np.argsort(eigenval)[::-1]
    eigenval = eigenval[indices]
    eigenvec = eigenvec[:, indices]
    # dimension reduction
    K = eigenvec[:, :n]
    L = np.diag(eigenval[:n])
    # calculating result
    Y = K @ L ** 1 / 2
    return np.real(Y)


def mds_sklearn(data):
    embedding = manifold.MDS(n_components=2)
    return embedding.fit_transform(data)
