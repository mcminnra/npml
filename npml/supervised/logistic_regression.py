import numpy as np
from npml.math.activations import sigmoid, softmax
from npml.util.metrics import log_loss


class BinaryLogisticRegression:
    """Classifier implementing binary logistic regression

    Uses a batch gradient descent optimization approach

    Parameters
    ----------
    fit_intercept : bool, optional (default = True)
        Fits an intercept (bias) term to the linear model

    learning_rate : float, optional (default = 5e-5)
        Learning rate for gradient descent

    optimization : str, optional "Adam" or None (default = None)
        Optimization method to use for gradient descent

        None   = Normal Gradient Descent Update
        "Adam" = Adam Optimization

    tol : float, optional (default = 1e-3)
        The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol).

    verbose : int, optional (default = 0)
        Any number larger than 0 prints actions verbosely

    Notes
    -----

    https://en.wikipedia.org/wiki/Logistic_regression
    """

    def __init__(self,
                 fit_intercept=True,
                 learning_rate=5e-5,
                 optimization=None,
                 tol=1e-3,
                 verbose=0):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.optimization = optimization
        self.tol = tol
        self.verbose = verbose

        self.weights = []

    def fit(self, features, target):
        """Fit model to training data

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
        if self.optimization == 'Adam':
            m_weights = np.zeros_like(self.weights)  # first-moment vector Adam Optimization for W1
            v_weights = np.zeros_like(self.weights)  # second-moment vector Adam Optimization for W1

        previous_loss = 100000  # Initialize Previous loss to a arbitrary high number to check for stopping
        stop = False
        iteration = 0

        while not stop:
            iteration += 1

            scores = np.dot(features, self.weights)
            predictions = sigmoid(scores)

            # Compute gradient
            # The gradient for Logistic Regression is the derivative of log loss (Negative Log-Likelihood)
            # Note: Calculating gradient for Binary and Multi-Class is different
            gradient = -np.dot(features.T, target - predictions)

            # Update Weights
            if self.optimization == 'Adam':
                # Constants for Adam Optimization (values are paper recommended values)
                beta1 = 0.9
                beta2 = 0.999
                eps = 1E-8
                # Update weights by using Adam Optimization (as opposed to simply learning_rate * gradient)
                # https://arxiv.org/pdf/1412.6980.pdf
                # http://cs231n.github.io/neural-networks-3/ (See Section: Per-parameter adaptive learning rate methods)
                m_weights = beta1 * m_weights + (1 - beta1) * gradient
                mt_weights = m_weights / (1 - beta1 ** iteration)
                v_weights = beta2 * v_weights + (1 - beta2) * (gradient ** 2)
                vt_weights = v_weights / (1 - beta2 ** iteration)
                self.weights += -self.learning_rate * mt_weights / (np.sqrt(vt_weights) + eps)
            else:
                self.weights -= self.learning_rate * gradient

            # Check to see if stopping criterion is reached
            loss = log_loss(features, target, self.weights)
            if loss > previous_loss - self.tol:
                stop = True
            else:
                previous_loss = loss

            # Verbose Output
            if self.verbose > 0:
                print("Iteration: {0}, Log Loss: {1}".format(iteration, loss))
                if stop:
                    print("Stopping Criterion Reached.")

        return self

    def predict(self, features):
        """Predict the class labels for the provided data

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

    optimization : str, optional "Adam" or None (default = None)
        Optimization method to use for gradient descent

        None   = Normal Gradient Descent Update
        "Adam" = Adam Optimization

    tol : float, optional (default = 1e-3)
        The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol).

    verbose : int, optional (default = 0)
        Any number larger than 0 prints actions verbosely

    Notes
    -----

    https://en.wikipedia.org/wiki/Multinomial_logistic_regression
    """


    def __init__(self,
                 fit_intercept=True,
                 learning_rate=5e-5,
                 optimization=None,
                 tol=1e-3,
                 verbose=0):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.optimization = optimization
        self.tol = tol
        self.verbose = verbose

        self.weights = []

    def fit(self, features, target):
        """Fit model to training data

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

        # Init weights
        self.weights = np.zeros([features.shape[1], len(np.unique(target))])
        if self.optimization == 'Adam':
            m_weights = np.zeros_like(self.weights)  # first-moment vector Adam Optimization for W1
            v_weights = np.zeros_like(self.weights)  # second-moment vector Adam Optimization for W1

        # Get number of training examples for the gradient descent update
        m = features.shape[0]

        # One-Hot encode target
        target_onehot = np.zeros((target.size, target.max() + 1))
        target_onehot[np.arange(target.size), target] = 1
        target = target_onehot

        previous_loss = 100000  # Initialize Previous loss to a arbitrary high number to check for stopping
        stop = False
        iteration = 0

        while not stop:
            iteration += 1

            scores = np.dot(features, self.weights)
            predictions = softmax(scores)

            # Compute gradient
            # The gradient for Logistic Regression is the derivative of log loss (Negative Log-Likelihood)
            # Note: Calculating gradient for Binary and Multi-Class is different
            gradient = (-1 / m) * np.dot(features.T, (target - predictions))

            # Update Weights
            if self.optimization == 'Adam':
                # Constants for Adam Optimization (values are paper recommended values)
                beta1 = 0.9
                beta2 = 0.999
                eps = 1E-8
                # Update weights by using Adam Optimization (as opposed to simply learning_rate * gradient)
                # https://arxiv.org/pdf/1412.6980.pdf
                # http://cs231n.github.io/neural-networks-3/ (See Section: Per-parameter adaptive learning rate methods)
                m_weights = beta1 * m_weights + (1 - beta1) * gradient
                mt_weights = m_weights / (1 - beta1 ** iteration)
                v_weights = beta2 * v_weights + (1 - beta2) * (gradient ** 2)
                vt_weights = v_weights / (1 - beta2 ** iteration)
                self.weights += -self.learning_rate * mt_weights / (np.sqrt(vt_weights) + eps)
            else:
                self.weights -= self.learning_rate * gradient

            # Check to see if stopping criterion is reached
            loss = log_loss(features, target, self.weights)
            if loss > previous_loss - self.tol:
                stop = True
            else:
                previous_loss = loss

            # Verbose Output
            if self.verbose > 0:
                print("Iteration: {0}, Log Loss: {1}".format(iteration, loss))
                if stop:
                    print("Stopping Criterion Reached.")

    def predict(self, features):
        """Predict the class labels for the provided data

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
