# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from npml.supervised import linear_regression


class TestLinearRegression(unittest.TestCase):
    def test_linear_regression_normal(self):
        # Load random blob dataset
        num_observations = 500
        data = np.random.multivariate_normal([0, 0], [[1, .9], [.9, 1]], num_observations)
        X = pd.DataFrame(np.vstack(data[:, 0]))
        y = pd.DataFrame(data[:, 1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Model
        lr = linear_regression.LinearRegression(optimization=None)
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        assert(isinstance(pred_lr, np.ndarray))

    def test_linear_regression_adam(self):
        # Load random blob dataset
        num_observations = 500
        data = np.random.multivariate_normal([0, 0], [[1, .9], [.9, 1]], num_observations)
        X = pd.DataFrame(np.vstack(data[:, 0]))
        y = pd.DataFrame(data[:, 1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Model
        lr = linear_regression.LinearRegression(optimization='adam')
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        assert(isinstance(pred_lr, np.ndarray))

    def test_linear_regression_radam(self):
        # Load random blob dataset
        num_observations = 500
        data = np.random.multivariate_normal([0, 0], [[1, .9], [.9, 1]], num_observations)
        X = pd.DataFrame(np.vstack(data[:, 0]))
        y = pd.DataFrame(data[:, 1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Model
        lr = linear_regression.LinearRegression(optimization='radam')
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        assert(isinstance(pred_lr, np.ndarray))

    def test_ridge_regression_normal(self):
        # Load random blob dataset
        num_observations = 500
        data = np.random.multivariate_normal([0, 0], [[1, .9], [.9, 1]], num_observations)
        X = pd.DataFrame(np.vstack(data[:, 0]))
        y = pd.DataFrame(data[:, 1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Model
        lr = linear_regression.RidgeRegression(optimization=None)
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        assert(isinstance(pred_lr, np.ndarray))

    def test_ridge_regression_adam(self):
        # Load random blob dataset
        num_observations = 500
        data = np.random.multivariate_normal([0, 0], [[1, .9], [.9, 1]], num_observations)
        X = pd.DataFrame(np.vstack(data[:, 0]))
        y = pd.DataFrame(data[:, 1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Model
        lr = linear_regression.RidgeRegression(optimization='adam')
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        assert(isinstance(pred_lr, np.ndarray))

    def test_ridge_regression_radam(self):
        # Load random blob dataset
        num_observations = 500
        data = np.random.multivariate_normal([0, 0], [[1, .9], [.9, 1]], num_observations)
        X = pd.DataFrame(np.vstack(data[:, 0]))
        y = pd.DataFrame(data[:, 1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Model
        lr = linear_regression.RidgeRegression(optimization='radam')
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        assert(isinstance(pred_lr, np.ndarray))
