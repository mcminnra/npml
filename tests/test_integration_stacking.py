# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from npml.util import stacking


class TestStacking(unittest.TestCase):
    def test_get_out_of_fold(self):
        # Generate a sample dataset
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=7, n_redundant=1, n_classes=2)

        # Train Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Init Model
        model = LogisticRegression(solver='lbfgs')

        X_train_oof, X_test_oof = stacking.get_out_of_fold(
            model, X_train, y_train, X_test, k=5, random_state=42)

        assert(isinstance(X_train_oof, np.ndarray))
        assert(isinstance(X_test_oof, np.ndarray))

    def test_stack(self):
        # Generate a sample dataset
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=7, n_redundant=1, n_classes=2)

        # Train Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # List of Models
        models = [
            LogisticRegression(solver='lbfgs'),
            RandomForestClassifier(n_estimators=100)
        ]

        # Get Stacked X
        X_train_l2, X_test_l2 = stacking.stack(models, X_train, y_train, X_test)

        assert(isinstance(X_train_l2, pd.DataFrame))
        assert(isinstance(X_test_l2, pd.DataFrame))


if __name__ == '__main__':
    unittest.main()
