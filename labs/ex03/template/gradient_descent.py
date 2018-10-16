# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.matmul(tx,w)
    return (-1/y.shape[0]) * np.matmul(tx.T, e)

def compute_loss_mse(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    y = np.array([y]).T.reshape([len(y), 1])
    w = np.array([w]).T.reshape([len(w), 1])
    
    e = y - np.matmul(tx,w) 
    
    return (1/(2 * y.shape[0])) * (e.T.dot(e))

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss_mse(y, tx, w)
        grad = compute_gradient(y, tx, w)
        w = w - (gamma * grad)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses[len(losses) - 1], w