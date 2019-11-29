# -*- coding: utf-8 -*-

import unittest
import numpy as np
from npml.math import activations


class TestActivations(unittest.TestCase):
    # Sigmoid
    def test_sigmoid_int(self):
        data = 1
        result = activations.sigmoid(data)
        np.testing.assert_almost_equal(result, 0.7310585786300049, decimal=14)

    def test_sigmoid_float(self):
        data = .85
        result = activations.sigmoid(data)
        np.testing.assert_almost_equal(result, 0.7005671424739729, decimal=14)

    def test_sigmoid_python_array(self):
        data = [1, .85]
        result = activations.sigmoid(data)
        np.testing.assert_array_almost_equal(
            result, [0.7310585786300049, 0.7005671424739729], decimal=14)

    def test_sigmoid_numpy_array(self):
        data = np.array([1, .85])
        result = activations.sigmoid(data)
        np.testing.assert_array_almost_equal(
            result,
            np.array([0.7310585786300049, 0.7005671424739729]),
            decimal=14)

    # Tanh
    def test_tanh_int(self):
        data = 1
        result = activations.tanh(data)
        np.testing.assert_almost_equal(result, 0.7615941559557649, decimal=14)

    def test_tanh_float(self):
        data = .85
        result = activations.tanh(data)
        np.testing.assert_almost_equal(result, 0.69106946983293049, decimal=14)

    def test_tanh_python_array(self):
        data = [1, .85]
        result = activations.tanh(data)
        np.testing.assert_array_almost_equal(
            result, [0.7615941559557649, 0.69106946983293049], decimal=14)

    def test_tanh_numpy_array(self):
        data = np.array([1, .85])
        result = activations.tanh(data)
        np.testing.assert_array_almost_equal(
            result,
            np.array([0.7615941559557649, 0.69106946983293049]),
            decimal=14)


if __name__ == '__main__':
    unittest.main()
