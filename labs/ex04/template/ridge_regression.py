# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    y = y.reshape([y.shape[0],1])
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(tx.T, tx) + 2 * len(y) * lambda_ * np.eye(tx.shape[1])), tx.T), y)
    e = y - np.matmul(tx, w)
    mse = (1/ (2 * y.shape[0])) * np.matmul(e.T, e)
    return (mse, w)
