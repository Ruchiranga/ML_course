# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import math

def compute_loss(y, tx, w):
    """ Compute the loss by MSE. """
    N = y.shape[0]
    e = y - tx.dot(w)
    MSE = (1/(2*N)) * (e.T).dot(e)
    return MSE

def compute_rmse_loss(y, tx, w):
    """ Compute the loss by MSE. """
    RMSE = math.sqrt(2 * compute_loss(y, tx, w))
    return RMSE