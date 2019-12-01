# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from npml.supervised import logistic_regression


class TestLogisticRegression(unittest.TestCase):
    def test_logistic_regression_normal(self):
        # Load random data
        X, y = datasets.make_classification(n_samples=500, n_features=5, n_classes=2)
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Model
        lr = logistic_regression.BinaryLogisticRegression(optimization=None)
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        assert(isinstance(pred_lr, np.ndarray))

    def test_logistic_regression_adam(self):
        # Load random data
        X, y = datasets.make_classification(n_samples=500, n_features=5, n_classes=2)
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Model
        lr = logistic_regression.BinaryLogisticRegression(optimization='adam')
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        assert(isinstance(pred_lr, np.ndarray))

    def test_logistic_regression_radam(self):
        # Load random data
        X, y = datasets.make_classification(n_samples=500, n_features=5, n_classes=2)
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Model
        lr = logistic_regression.BinaryLogisticRegression(optimization='radam')
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        assert(isinstance(pred_lr, np.ndarray))

    def test_softmax_regression_normal(self):
        # Load random data
        X, y = datasets.make_classification(
            n_samples=500, n_features=10, n_classes=3, n_informative=4)
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Model
        lr = logistic_regression.SoftmaxRegression(optimization=None)
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        assert(isinstance(pred_lr, np.ndarray))

    def test_softmax_regression_adam(self):
        # Load random data
        X, y = datasets.make_classification(
            n_samples=500, n_features=10, n_classes=3, n_informative=4)
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Model
        lr = logistic_regression.SoftmaxRegression(optimization='adam')
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        assert(isinstance(pred_lr, np.ndarray))

    def test_softmax_regression_radam(self):
        # Load random data
        X, y = datasets.make_classification(
            n_samples=500, n_features=10, n_classes=3, n_informative=4)
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Model
        lr = logistic_regression.SoftmaxRegression(optimization='radam')
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        assert(isinstance(pred_lr, np.ndarray))
