# -*- coding: utf-8 -*-

import numpy as np
from npml.math.activations import sigmoid, softmax
from npml.util.metrics import log_loss


class BinaryLogisticRegression:
    """
    Classifier implementing binary logistic regression

    Uses a batch gradient descent optimization approach

    Parameters
    ----------
    fit_intercept : bool, optional (default = True)
        Fits an intercept (bias) term to the linear model

    learning_rate : float, optional (default = 5e-5)
        Learning rate for gradient descent

    optimization : str, optional "adam", 'radam', or None (default = None)
        Optimization method to use for gradient descent

        None   = Normal Gradient Descent Update
        "adam" = Adam Optimization
        "radam" = Rectified Adam Optimization

    tol : float, optional (default = 1e-3)
        The stopping criterion. If it is not None, the iterations will stop when
        (loss > previous_loss - tol).

    verbose : int, optional (default = 0)
        Any number larger than 0 prints actions verbosely

    output_freq: int, optional (default = 10000)
        The iteration frequency to print an error message. Only displays if verbose is greater
        than 0.

    Notes
    -----
    https://en.wikipedia.org/wiki/Logistic_regression
    """

    def __init__(self,
                 fit_intercept=True,
                 learning_rate=5e-5,
                 optimization=None,
                 tol=1e-3,
                 verbose=0,
                 output_freq=10000):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.optimization = optimization.lower() if isinstance(optimization, str) else optimization
        self.tol = tol
        self.verbose = verbose
        self.output_freq = output_freq

        self.weights = []

    def fit(self, features, target):
        """
        Fit model to training data

        Parameters
        ----------
        features : array-like, shape (n_samples, n_features)
            features matrix

        target : array-like, shape [n_samples]
            target array
        """

        if not type(target) == np.dtype:
            try:
                # try to convert to ndarray
                target = target.values.ravel()
            except Exception:
                print("Error - Couldn't convert to ndarray")
                raise

        if self.fit_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))

        # Initialize weights
        self.weights = np.zeros(features.shape[1])
        if self.optimization == 'adam':
            # Constants for Adam Optimization (values are paper recommended values)
            beta1 = 0.9
            beta2 = 0.999
            eps = 1E-8
            # first-moment vector Adam Optimization for W1
            m = np.zeros_like(self.weights)
            # second-moment vector Adam Optimization for W1
            v = np.zeros_like(self.weights)
        elif self.optimization == 'radam':
            # Constants for Rectified Adam Optimization (values are paper recommended values)
            beta1 = 0.9
            beta2 = 0.999
            eps = 1E-8
            # first-moment vector Rectified Adam Optimization for W1
            m = np.zeros_like(self.weights)
            # second-moment vector Recified Adam Optimization for W1
            v = np.zeros_like(self.weights)
            # Maximum length of approximated SMA
            p_inf = 2 / ((1-beta2)-1)

        # Print settings if verbose
        if self.verbose > 0:
            print(f'Optimization: {self.optimization}')

        # Initialize Previous loss to a arbitrary high number to check for stopping
        previous_loss = 100000
        stop = False
        iteration = 0

        while not stop:
            iteration += 1

            scores = np.dot(features, self.weights)
            predictions = sigmoid(scores)

            # Compute gradient
            # The gradient for Logistic Regression is the derivative of log loss
            # (Negative Log-Likelihood)
            # Note: Calculating gradient for Binary and Multi-Class is different
            gradient = -np.dot(features.T, target - predictions)

            # Update Weights
            if self.optimization == 'adam':
                # Update weights by using Adam Optimization
                # (as opposed to simply learning_rate * gradient)
                # https://arxiv.org/pdf/1412.6980.pdf
                # http://cs231n.github.io/neural-networks-3/
                # (See Section: Per-parameter adaptive learning rate methods)
                m = beta1 * m + (1 - beta1) * gradient
                mt = m / (1 - beta1 ** iteration)
                v = beta2 * v + (1 - beta2) * (gradient ** 2)
                vt = v / (1 - beta2 ** iteration)
                self.weights += -self.learning_rate * mt / (np.sqrt(vt) + eps)
            elif self.optimization == 'radam':
                # Update exponetial 1st and 2nd moment
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient ** 2)
                # Compute bias-corrected moving average
                mt = m / (1 - beta1 ** iteration)
                # Compute length of the approximated SMA
                pt = p_inf - ((2 * iteration * (beta2 ** iteration)) / (1 - (beta2 ** iteration)))
                # if variance is tractable, updated with adapative momentum, else unadapated
                if pt > 4:
                    # Compute bias-corrected moving 2nd moment
                    vt = v / (1 - beta2 ** iteration)
                    # Compute variance rectification term
                    rt = np.sqrt(((pt-4)(pt-2)(p_inf))/((p_inf-4)(p_inf-2)(pt)))
                    # Update weights with adaptive momentum
                    self.weights += -self.learning_rate * rt * mt / vt
                else:
                    # Update weights with un-adapted momentum
                    self.weights += -self.learning_rate * mt
            else:
                self.weights -= self.learning_rate * gradient

            # Check to see if stopping criterion is reached
            loss = log_loss(features, target, self.weights)
            if loss > previous_loss - self.tol:
                stop = True
            else:
                previous_loss = loss

            # Verbose Output
            if self.verbose > 0 and iteration % self.output_freq == 0:
                print("Iteration: {0}, Log Loss: {1}".format(iteration, loss))
                if stop:
                    print("Stopping Criterion Reached.")

        return self

    def predict(self, features):
        """
        Predict the class labels for the provided data

        Parameters
        ----------
        features : array-like, shape (n_samples, n_features)

        Returns
        -------
        target : array of shape [n_samples]
            Class labels for each data sample.
        """

        if self.fit_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))

        final_scores = np.dot(features, self.weights)
        predictions = np.round(sigmoid(final_scores))

        return predictions


class SoftmaxRegression:
    """
    Softmax (Multi-Class) Logistic Regression

    Uses a gradient descent optimization approach

    Parameters
    ----------
    fit_intercept : bool, optional (default = True)
        Fits an intercept (bias) term to the linear model

    learning_rate : float, optional (default = 5e-5)
        Learning rate for gradient descent

    optimization : str, optional "adam", 'radam', or None (default = None)
        Optimization method to use for gradient descent

        None   = Normal Gradient Descent Update
        "adam" = Adam Optimization
        "radam" = Rectified Adam Optimization

    tol : float, optional (default = 1e-3)
        The stopping criterion. If it is not None, the iterations will stop when
        (loss > previous_loss - tol).

    verbose : int, optional (default = 0)
        Any number larger than 0 prints actions verbosely

    output_freq: int, optional (default = 10000)
        The iteration frequency to print an error message. Only displays if verbose is greater
        than 0.

    Notes
    -----
    https://en.wikipedia.org/wiki/Multinomial_logistic_regression
    """

    def __init__(self,
                 fit_intercept=True,
                 learning_rate=5e-5,
                 optimization=None,
                 tol=1e-3,
                 verbose=0,
                 output_freq=10000):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.optimization = optimization.lower() if isinstance(optimization, str) else optimization
        self.tol = tol
        self.verbose = verbose
        self.output_freq = output_freq

        self.weights = []

    def fit(self, features, target):
        """
        Fit model to training data

        Parameters
        ----------
        features : array-like, shape (n_samples, n_features)
            features matrix

        target : array-like, shape [n_samples]
            target array
        """

        if not type(target) == np.dtype:
            try:
                # try to convert to ndarray
                target = target.values.ravel()
            except Exception:
                print("Error - Couldn't convert to ndarray")
                raise

        if self.fit_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))

        # Get number of training examples for the gradient descent update
        num_rows = features.shape[0]

        # Init weights
        self.weights = np.zeros([features.shape[1], len(np.unique(target))])
        if self.optimization == 'adam':
            # Constants for Adam Optimization (values are paper recommended values)
            beta1 = 0.9
            beta2 = 0.999
            eps = 1E-8
            # first-moment vector Adam Optimization for W1
            m = np.zeros_like(self.weights)
            # second-moment vector Adam Optimization for W1
            v = np.zeros_like(self.weights)
        elif self.optimization == 'radam':
            # Constants for Rectified Adam Optimization (values are paper recommended values)
            beta1 = 0.9
            beta2 = 0.999
            eps = 1E-8
            # first-moment vector Rectified Adam Optimization for W1
            m = np.zeros_like(self.weights)
            # second-moment vector Recified Adam Optimization for W1
            v = np.zeros_like(self.weights)
            # Maximum length of approximated SMA
            p_inf = 2 / ((1-beta2)-1)

        # Print settings if verbose
        if self.verbose > 0:
            print(f'Optimization: {self.optimization}')

        # One-Hot encode target
        target_onehot = np.zeros((target.size, target.max() + 1))
        target_onehot[np.arange(target.size), target] = 1
        target = target_onehot

        # Initialize Previous loss to a arbitrary high number to check for stopping
        previous_loss = 100000
        stop = False
        iteration = 0

        while not stop:
            iteration += 1

            scores = np.dot(features, self.weights)
            predictions = softmax(scores)

            # Compute gradient
            # The gradient for Logistic Regression is the derivative of log loss
            # (Negative Log-Likelihood)
            # Note: Calculating gradient for Binary and Multi-Class is different
            gradient = (-1 / num_rows) * np.dot(features.T, (target - predictions))

            # Update Weights
            # Update Weights
            if self.optimization == 'adam':
                # Update weights by using Adam Optimization
                # (as opposed to simply learning_rate * gradient)
                # https://arxiv.org/pdf/1412.6980.pdf
                # http://cs231n.github.io/neural-networks-3/
                # (See Section: Per-parameter adaptive learning rate methods)
                m = beta1 * m + (1 - beta1) * gradient
                mt = m / (1 - beta1 ** iteration)
                v = beta2 * v + (1 - beta2) * (gradient ** 2)
                vt = v / (1 - beta2 ** iteration)
                self.weights += -self.learning_rate * mt / (np.sqrt(vt) + eps)
            elif self.optimization == 'radam':
                # Update expoential 1st and 2nd moment
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient ** 2)
                # Compute bias-corrected moving average
                mt = m / (1 - beta1 ** iteration)
                # Compute length of the approximated SMA
                pt = p_inf - ((2 * iteration * (beta2 ** iteration)) / (1 - (beta2 ** iteration)))
                # if variance is tractable, updated with adapative momentum, else unadapated
                if pt > 4:
                    # Compute bias-corrected moving 2nd moment
                    vt = v / (1 - beta2 ** iteration)
                    # Compute variance rectification term
                    rt = np.sqrt(((pt-4)(pt-2)(p_inf))/((p_inf-4)(p_inf-2)(pt)))
                    # Update weights with adaptive momentum
                    self.weights += -self.learning_rate * rt * mt / vt
                else:
                    # Update weights with un-adapted momentum
                    self.weights += -self.learning_rate * mt
            else:
                self.weights -= self.learning_rate * gradient

            # Check to see if stopping criterion is reached
            loss = log_loss(features, target, self.weights)
            if loss > previous_loss - self.tol:
                stop = True
            else:
                previous_loss = loss

            # Verbose Output
            if self.verbose > 0 and iteration % self.output_freq == 0:
                print("Iteration: {0}, Log Loss: {1}".format(iteration, loss))
                if stop:
                    print("Stopping Criterion Reached.")

    def predict(self, features):
        """
        Predict the class labels for the provided data

        Parameters
        ----------
        features : array-like, shape (n_samples, n_features)

        Returns
        -------
        target : array of shape [n_samples]
            Class labels for each data sample.
        """

        if self.fit_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))

        final_scores = np.dot(features, self.weights)
        predictions_one_hot = softmax(final_scores)
        predictions = np.apply_along_axis(np.argmax, 1, predictions_one_hot)

        return predictions
