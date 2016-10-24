from models.nade import BinaryNade
from keras import backend as K
import numpy as np
import theano
import unittest

__author__ = 'theopavlakou'

rng = np.random
np.set_printoptions(precision=4)


def vec_to_num(x):
    """
    Change a vector binary representation of a number into an integer.
    :param x:   (N, D) or (D, ) numpy array with 1s and 0s. D is the dimension
                of the data and N is the number of data points. The 0th bit is
                the most significant bit.
    :return:    The base 10 representation.
    """
    s = 0
    if len(x.shape) == 1:
        l = len(x)
        for i in xrange(l):
            s += x[i]*2**(l-i-1)
    else:
        l = x.shape[1]
        for i in xrange(l):
            s += x[:, i]*2**(l-i-1)
    return s


def sample(N):
    """
    Returns a vector of dimension 3 which has been sampled from the following distribution:
    p(x_1, x_2, x_3) = p(x_1)p(x_2)p(x_3|x_1, x_2) i.e. 2 parents.
    :param: N - Number of samples
    :return: an (N, 3) numpy array
    """
    p_x_0 = 0.75
    p_x_1 = 0.25
    p_x_3_cond = np.array([0.5, 0.75, 0.25, 0.1])
    X = np.zeros((N, 3), dtype=int)
    X[:, 0] = (rng.rand(N) < p_x_0).astype(int)
    X[:, 1] = (rng.rand(N) < p_x_1).astype(int)
    X[:, 2] = (rng.rand(N) < p_x_3_cond[vec_to_num(X[:, 0:-1])]).astype(int)
    return X


class TestNade(unittest.TestCase):

    def test_works(self):
        ####################################
        # Load the data
        ####################################
        training_epoch_size = 100000

        X_train = sample(training_epoch_size).astype(np.float32)
        np.bincount(vec_to_num(X_train.astype(int)))

        ####################################
        # Set up some variables
        ####################################
        D = X_train.shape[1]
        H = 20
        minibatch_size = 25

        ####################################
        # Train the network
        ####################################
        nade = BinaryNade(D, H)
        nade.initialise()
        nade.compile('adam', loss=nade.cost)
        temp = K.exp((nade.input*K.log(nade.output) + (1-nade.input)*K.log(1-nade.output)).sum(1))
        probability_x = theano.function([nade.input], temp, allow_input_downcast=True)

        X = np.array([[0, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [0, 1, 1],
                      [1, 0, 0],
                      [1, 0, 1],
                      [1, 1, 0],
                      [1, 1, 1]])
        n_samples = 100000
        X_sampled = nade.sample(n_samples).astype(int)

        print("\n-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------- Before training ------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")

        sample_dist = np.bincount(vec_to_num(X_train.astype(int)))/np.float(X_train.shape[0])
        print("  The sample distribution is:             {0}"
                   .format(sample_dist))
        print("  Probability distribution given by NADE: {0}"
                   .format(probability_x(X)))
        print("  The SAMPLED empirical distribution is:  {0}"
                   .format(np.bincount(vec_to_num(X_sampled))/np.float(X_sampled.shape[0])))
        print("  This adds up to: {0}".format(probability_x(X).sum()))
        print("")

        self.assertLess(abs(1 - probability_x(X).sum()), 0.0001)
        nade.fit(X_train, X_train, batch_size=minibatch_size, nb_epoch=10)

        X_sampled = nade.sample(n_samples).astype(int)
        print("\n-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------- After training -------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")

        print("  The sample distribution is:             {0}"
                   .format(np.bincount(vec_to_num(X_train.astype(int)))/np.float(X_train.shape[0])))
        print("  Probability distribution given by MADE: {0}"
                   .format(probability_x(X)))
        print("  The SAMPLED empirical distribution is:  {0}"
                   .format(np.bincount(vec_to_num(X_sampled))/np.float(X_sampled.shape[0])))
        print("  This adds up to: {0}".format(probability_x(X).sum()))
        self.assertLess(abs(1 - probability_x(X).sum()), 0.0001)

        ####################################
        # Record costs
        ####################################
        print("")
        print("\n=============================================")
        print("===== Final cost for NADE {0} =====".format(nade.evaluate(X_train, X_train)))
        print("=============================================\n")

        entropy = sum([-a*np.log(a) for a in sample_dist])
        print("Entropy of distribution is: {0}".format(entropy))
        self.assertLess(abs(entropy - nade.evaluate(X_train, X_train)), 0.01)
