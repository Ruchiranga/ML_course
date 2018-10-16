# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    num_samples = x.shape[0]
    tx = np.ones(num_samples)
    for i in range(1, (degree+1)):
        tx = np.c_[tx, np.power(x, i)]
    return tx
