# -*- coding: utf-8 -*-
"""A function to compute the cost."""
import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - np.matmul(tx,w)
    return (1/(2 * y.shape[0])) * (e.T.dot(e))
