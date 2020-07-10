#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import settings
import numpy as np
import pandas as pd
from sklearn import preprocessing


def mae(y_true, y_pred):
    """ Compute Mean Absolute Error

    This function computes MAE on the non log
    error

    Parameters
    ----------
        y_true : list(float)
            True value for
            a given sample of data
        y_pred : list(float)
            Predicted value for
            a given sample of data

    Returns
    -------
        MAE : float
            Mean Absolute Error

    """
    y_pred = np.array([10 ** -y for y in y_pred])
    y_true = np.array([10 ** -y for y in y_true])
    return np.mean(np.abs(y_pred - y_true))


def is_dominant(x, y):
    """ Checks if the configuration x is dominant over y

    Parameters
    ----------
        x : list(float)
            configuration
        y : list(float)
            configuration

    Returns
    -------
        Dominance Truth Value : bool
            True if x is dominant over y, False otherwise

    """
    n = len(x) if isinstance(x, list) else x.shape[0]
    return all([x[i] > y[i] for i in range(n)])


def couples(precision):
    """ Counts number of couples dominant dominated """
    n = len(precision)
    couples = []
    for i in range(n):
        x = np.repeat([precision[i]], n, axis=0)
        dominated_idx = np.where(np.all(x > precision, axis=1))[0]
        couples += [(i, j) for j in list(dominated_idx)]

    return couples


def violated_const(precision, error):
    """ Counts number of violated_const

        if x' is dominant on x'' -> -log10(e(x')) > -log10(e(x''))

    """
    n = len(precision)
    violated_const = [(i, j) for (i, j) in couples(precision) if error[i] < error[j]]

    return violated_const


def duplicates(error):
    """ Computes the number of duplicates in the error predicted, especially,
        sums the number of repetition of the 3 most frequent elements.

        This function is used to check the validity of the results predicted
        by the model. As observed previous experimens, high values in the
        multiplier lead to trivial prediction, i.e. for every instance the prediction
        has often the same outcome
    """
    u, c = np.unique(np.round(error, 5), return_counts=True)
    dup = list(zip(u, c))
    dup.sort(key=lambda x: x[1])
    return sum([dup[-1][1], dup[-2][1], dup[-3][1]]) / len(error)
