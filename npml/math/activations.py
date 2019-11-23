# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import expit  # pylint: disable=no-name-in-module


def sigmoid(x, derivative=False):
    """Calculates the Sigmoid function

    .. math::S(x)={\frac {1}{1+e^{-x}}}={\frac {e^{x}}{e^{x}+1}}

    Parameters
    ----------
    x : int, float, or array_like
    derivative : bool
        Calculates the derivative of Sigmoid instead

    Returns
    -------
    int, float, or array_like
        Output of the Sigmoid function
    """
    return x * (1 - x) if derivative else expit(x)


def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2


def tanh(x, derivative=False):
    """Calculates the Hyperbolic Tangent (tanh) function

    .. math::\tanh x={\frac {e^{2x}-1}{e^{2x}+1}}

    Parameters
    ----------
    x : int, float, or array_like
    derivative : bool
        Calculates the derivative of Hyperbolic Tangent instead

    Returns
    -------
    int, float, or array_like
        Output of the Hyperbolic Tangent function
    """
    return 1 - np.power(x, 2) if derivative else np.tanh(x)
