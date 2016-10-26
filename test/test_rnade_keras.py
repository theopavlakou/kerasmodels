from __future__ import division
from models.nade.nade_keras import RNade
import numpy as np
import unittest

__author__ = 'theopavlakou'

rng = np.random
np.set_printoptions(precision=4)


def sample(N):
    """
    Returns a vector of dimension 2 which has been sampled from a mixture of Gaussians
    with the following parameters:
        mean = [[1, 1], [-1, -1]]
        Covariances = [[[0.5, 0],
                        [0, 0.5]],
                        [[0.5, 0],
                        [0, 0.5]]]
        mixing coefficients = [0.25, 0.75]
    :param: N - Number of samples
    :return: an (N, 3) numpy array
    """
    X_1 = rng.multivariate_normal([1, 1], [[0.5, 0], [0, 0.5]], N)
    X_2 = rng.multivariate_normal([-1, -1], [[0.5, 0], [0, 0.5]], N)
    p = (rng.random((N, 1)) < 0.25).astype(int)
    return X_1*p + X_2*(1-p)


def log_probability_density_gaussian(X, mu, Sigma):
    """
    Returns the log probability density of a multivariate Gaussian with
    mean mu and covariance matrix Sigma.
    :param X:   (N, D) numpy array of data, with N being the number of
                examples and D being the dimension of the Gaussian.
    :param mu:  (D, ) numpy array: the mean of the Gaussian.
    :param Sigma: (D, D) numpy array: the covariance matrix.
    :return:    (N, ) numpy array with the log probability density of each
                example.
    """
    X = X - mu
    temp = np.linalg.solve(Sigma, X.T).T
    return -.5*(X*temp).sum(axis=1) - X.shape[1]/2*np.log(2*np.pi) - .5*np.linalg.slogdet(Sigma)[1]


def differential_entropy_under_mogs(X):
    """
    Gets the (sample) differential entropy of the data under the mixture
    of Gaussians defined in the function sample. This is given as the
    mean of the negative log probability of the data under the mixture
    of Gaussians' distribution.

    :param X:   (N, D) numpy array of data, with N being the number of
                examples and D being the dimension of the Gaussian.
    :return: a scalar.
    """
    p_x_1 = np.exp(log_probability_density_gaussian(X, np.array([1, 1]), np.array([[0.5, 0], [0, 0.5]])))
    p_x_2 = np.exp(log_probability_density_gaussian(X, np.array([-1, -1]), np.array([[0.5, 0], [0, 0.5]])))
    p_x = 0.25*p_x_1 + 0.75*p_x_2
    return -np.log(p_x).mean()


class TestRNade(unittest.TestCase):

    def test_works(self):
        ####################################
        # Load the data
        ####################################
        training_epoch_size = 100000
        X_train = sample(training_epoch_size).astype(np.float32)

        ####################################
        # Set up some variables
        ####################################
        D = X_train.shape[1]
        H = 20
        K = 2
        minibatch_size = 25

        ####################################
        # Train the network
        ####################################
        rnade = RNade(D, H, K)
        rnade.compile('adam', loss=rnade.cost)
        rnade.fit(X_train, X_train, batch_size=minibatch_size, nb_epoch=10)

        ####################################
        # Record costs
        ####################################
        print("")
        print("\n=============================================")
        print("===== Final cost for NADE {0} =====".format(rnade.evaluate(X_train, X_train)))
        print("=============================================\n")
        entropy = differential_entropy_under_mogs(X_train)
        print("Entropy of distribution is: {0}".format(entropy))
        self.assertLess(abs(entropy - rnade.evaluate(X_train, X_train)), 0.01)