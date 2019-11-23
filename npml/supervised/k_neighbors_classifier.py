import numpy as np

class KNeighborsClassifier:
    """Classifier implementing the k-nearest neighbors vote.

    Implements a brute force approach

    Parameters
    ----------
    neighbors : int, optional (default = 5)
        Number of neighbors to vote for new test sample

    Notes
    -----

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """

    def __init__(self, neighbors=5):
        self.neighbors = neighbors

    def fit(self, X_train, y_train):
        """Fit model to training data

        Basically saves the train data to the class to be used by predict

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            features matrix

        y_train : array-like, shape [n_samples]
            target array
        """

        self.X_train = X_train

        if not type(y_train) == np.dtype:
            try:
                # try to convert to ndarray
                self.y_train = y_train.values.ravel()
            except:
                raise Exception("Error - Couldn't convert to ndarray")

        return self

    def predict(self, X_test):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_test : array of shape [n_samples]
            Class labels for each data sample.
        """

        predictions = []

        for xtest in X_test.values:
            nearest = []

            for i, xtrain in enumerate(self.X_train.values):

                # Euclidean distance
                distance = np.linalg.norm(xtest - xtrain)

                if len(nearest) == self.neighbors:
                    # check to see if biggest value in nearest is more than current distance
                    if nearest[-1][0] > distance:
                        nearest[-1] = (distance, self.y_train[i])
                elif len(nearest) < self.neighbors:
                    nearest.append((distance, self.y_train[i]))
                    nearest.sort(key=lambda x: x[0])  # sorts by first number in tuple
                else:
                    raise ValueError("Error - More in nearest arr than passed neighbors var")

            # append most common occuring class in nearest
            predictions.append(max(set(nearest), key=nearest.count)[1])

        y_test = np.array(predictions)

        return y_test
