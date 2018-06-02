# -*- coding: utf-8 -*-
# svm.py: Riemannian kernel for support vector machines
# author : Antoine Passemiers

import numpy as np
import scipy.linalg
from sklearn.datasets import make_classification
from sklearn.svm import SVC


CREF = np.random.rand(32, 32)


def hlmap(Cp, sqrtCinv):
    """ Locally project a SPD matrix onto the plane tangent to
    matrix CREF, whose inverse square root is provided as second argument.

    Args:
        Cp (np.ndarray):
            SPD matrix to be projected onto the tangent plane
        sqrtCinv (np.ndarray):
            Inverse square root of the reference SPD matrix
    """
    return scipy.linalg.logm(np.dot(sqrtCinv, np.dot(Cp, sqrtCinv)))


def riemannian_kernel(X, Y):
    """ Riemannian-based kernel for symmetric and positive definite (SPD) matrices.

    Returns an array of shape (n_samples, n_samples) where element with
    index (i, j) is the product between samples i and j in the local space
    tangent to the manifold at point CREF.

    Args:
        X (np.ndarray):
            Sample matrix of shape (n_samples, n_features)
        Y (np.ndarray):
            Sample matrix of shape (n_samples, n_features)
    """
    print(X.shape, Y.shape)
    dtype = np.float16
    n_samples, n_features = X.shape
    n_original_variables = int(np.sqrt(n_features))

    # Reshape vectorized SPD matrices to actual 2D matrices
    CS = X.reshape(n_samples, n_original_variables, n_original_variables)

    # Project each sample from the manifold to the local tangent space
    mapped_cs = [hlmap(C, CREF).astype(dtype) for C in CS]

    # Compute scalar product in the tangent space for each pair
    # of SPD matrices and return the resulting kernel matrix
    K = np.empty((n_samples, n_samples), dtype=dtype)
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.trace(np.dot(mapped_cs[i], mapped_cs[j]))
    return K


if __name__ == "__main__":
    n_electrodes = 32
    X, y = make_classification(
        n_samples=100, n_features=n_electrodes**2, n_classes=2)

    svm = SVC(C=1.0, kernel=riemannian_kernel)
    svm.fit(X, y)
