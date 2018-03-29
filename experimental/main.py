# -*- coding: utf-8 -*-
# cov.py
# author : Antoine Passemiers

import numpy as np
import matplotlib.pyplot as plt


def extract_cov_matrices(data, w):
    n_features = data.shape[1]
    sigmas = np.empty((data.shape[0], n_features, n_features), dtype=np.float)
    means = np.asarray(np.mean(np.asarray(data)[:w, :], axis=0), dtype=np.float)
    last_sigma = np.cov(np.asarray(data)[:w, :].T)
    np_sigmas = np.asarray(sigmas)
    np_sigmas[:w, :, :] = np.repeat(np.asarray(last_sigma).reshape((1, n_features, n_features), order='C'), w, axis=0)
    for i in range(w, data.shape[0]):
        np_sigmas[i, :, :] = np.cov(np.asarray(data)[i-w:i].T)
    return np.asarray(np_sigmas)


def inc_extract_cov_matrices(data, w):
    n_features = data.shape[1]
    sigmas = np.empty((data.shape[0], n_features, n_features), dtype=np.float)
    means = np.asarray(np.mean(np.asarray(data)[:w, :], axis=0), dtype=np.float)
    old_means = np.copy(means)
    last_sigma = np.cov(np.asarray(data)[:w, :].T)
    np_sigmas = np.asarray(sigmas)
    np_sigmas[:w, :, :] = np.repeat(np.asarray(last_sigma).reshape((1, n_features, n_features), order='C'), w, axis=0)
    for i in range(w, np_sigmas.shape[0]):
        old_means[:] = means[:]
        means += (data[i, :] - data[i-w, :]) / w
        for a in range(n_features):
            for b in range(a+1):
                c = np_sigmas[i-1, a, b]
                c -= (data[i-w, a] * data[i-w, b]) / w
                c += (data[i, a] * data[i, b]) / w
                c += old_means[a] * old_means[b] - means[a] * means[b]
                np_sigmas[i, a, b] = np_sigmas[i, b, a] = c
    return np.asarray(np_sigmas)


data = np.random.rand(800, 32)
cov = extract_cov_matrices(data, 50)
cov2 = inc_extract_cov_matrices(data, 50)

print(cov.shape)
print(np.isclose(cov, cov2).sum(), cov.shape[0] * cov.shape[1] * cov.shape[2])

plt.plot(cov[:, 4, 8])
plt.plot(cov2[:, 4, 8])
plt.show()