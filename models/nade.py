from __future__ import division
from keras.models import Sequential, Model
from keras.engine.topology import Layer
from keras.layers import Input
from keras.initializations import glorot_normal
from keras.layers import Activation
from keras import backend as K
from theano import tensor as T
import numpy as np
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from common_methods.vector_methods import logsumexp


class NadeInputLayer(Layer):
    """
    A class that implements the NADE input-to-hidden layer as illustrated in
    http://homepages.inf.ed.ac.uk/imurray2/pub/11nade/. This runs in O(HD) time,
    where H is the number of hidden units and D is the input dimension.
    """
    def __init__(self, output_dim, input_dim, weight_init=glorot_normal, **kwargs):
        """
        This should always be the input layer, so we need to provide the
        input dimension.

        :param output_dim: The dimension of the hidden layer in NADE.
        :param input_dim: The dimension of the input data.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_init = weight_init
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(NadeInputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        # The first column of W is c in the paper (the bias term), since the last column of
        # W is never used in the original paper.
        self.W = self.weight_init((self.input_dim, self.output_dim), name="W_input_to_hidden")
        self.rho = self.weight_init((self.input_dim,), name="rho_activation_multiplier")
        self.trainable_weights = [self.W, self.rho]
        super(NadeInputLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        activation = self.W.T * T.horizontal_stack(T.ones((x.shape[0], 1)), x[:, 0:-1]).dimshuffle(0, 'x', 1)
        # The rho parameter was not used in the original paper, but was in
        # https://arxiv.org/pdf/1306.0186v2.pdf.
        return self.rho * T.extra_ops.cumsum(activation, 2)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim


class NadeOutputBinaryLayer(Layer):
    """
    A class that implements a hidden-to-output layer for NADE for binary data.
    This runs in O(HD) time also.
    """

    def __init__(self, output_dim, weight_init=glorot_normal, **kwargs):
        self.output_dim = output_dim
        self.weight_init = weight_init
        super(NadeOutputBinaryLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.V = self.weight_init((input_dim, self.output_dim), name="V_hidden_to_output")
        self.b = self.weight_init((self.output_dim,), name="b_output")
        self.trainable_weights = [self.V, self.b]
        super(NadeOutputBinaryLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        return (self.V * x).sum(axis=1) + self.b

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim


class RNadeOutputLayer(Layer):
    """
    A class that implements a hidden-to-output layer for RNADE, as outlined in
    https://arxiv.org/abs/1306.0186. This runs in O(HKD) time.
    """
    def __init__(self, output_dim, number_components_per_conditional, input_var, weight_init=glorot_normal, **kwargs):
        self.output_dim = output_dim
        self.num_components = number_components_per_conditional
        self.weight_init = weight_init
        self.input_var = input_var
        super(RNadeOutputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.V_alpha = self.weight_init((self.num_components, input_dim, self.output_dim), name="V_alpha")
        self.V_mu = self.weight_init((self.num_components, input_dim, self.output_dim), name="V_mu")
        self.V_sigma = self.weight_init((self.num_components, input_dim, self.output_dim), name="V_sigma")
        self.b_alpha = self.weight_init((self.num_components, self.output_dim), name="b_alpha")
        self.b_mu = self.weight_init((self.num_components, self.output_dim), name="b_mu")
        self.b_sigma = self.weight_init((self.num_components, self.output_dim), name="b_sigma")

        self.trainable_weights = [self.V_alpha, self.V_mu, self.V_sigma,
                                  self.b_alpha, self.b_mu, self.b_sigma]

        super(RNadeOutputLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def get_log_alpha(self, x):
        # (N, 1, H, D)
        h_new_shape = x.dimshuffle(0, 'x', 1, 2)
        # ((N, 1, H, D) (1, K, H, D)).sum(axis=2) => (N, K, D)
        activation_alpha = (h_new_shape * self.V_alpha).sum(2)
        # (N, K, D). Parameters to the mixture of Gaussians for each dimension.
        exponent_alpha = self.b_alpha + activation_alpha
        # We subtract the max because by doing so we avoid overflows and the ratio for softmax is left invariant.
        exponent_alpha_minus_max = exponent_alpha - exponent_alpha.max(axis=1).dimshuffle(0, 'x', 1)
        # (N, K, D) <= (N, K, D) - (N, 1, D)
        return exponent_alpha_minus_max - K.log(K.sum(K.exp(exponent_alpha_minus_max), axis=1)).dimshuffle(0, 'x', 1)

    def get_mu(self, x):
        h_new_shape = x.dimshuffle(0, 'x', 1, 2)
        activation_mu = K.sum(h_new_shape * self.V_mu, axis=2)
        # (N, K, D). Parameters to the mixture of Gaussians for each dimension.
        return self.b_mu + activation_mu

    def get_log_sigma(self, x):
        h_new_shape = x.dimshuffle(0, 'x', 1, 2)
        activation_sigma = K.sum(h_new_shape * self.V_sigma, axis=2) + self.b_sigma
        # We clip to ensure that we don't get numerical errors.
        return K.clip(activation_sigma, -20, 20)

    def call(self, x, mask=None):
        log_alpha = self.get_log_alpha(x)
        mu = self.get_mu(x)
        log_sigma = self.get_log_sigma(x)

        sigma = K.exp(log_sigma)

        # (N, K, D)
        log_p_z_x_joint = (-1/(2*K.square(sigma))*K.square(self.input_var.dimshuffle(0, 'x', 1) - mu) + log_alpha -
                          1/2*K.log(2*np.pi) - log_sigma)

        return K.sum(logsumexp(log_p_z_x_joint, axis=1), axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim


class RNade(Model):

    def __init__(self, input_dim, hidden_dim, number_components_per_conditional, weight_init=glorot_normal,
                 nonlinearity=K.relu, name="RNade"):
        """
        Initialise RNADE, as is outlined in the following paper:
        https://arxiv.org/abs/1306.0186

        :param input_dim: The input dimensionality.
        :param hidden_dim: The hidden layer dimensionality.
        :param number_components_per_conditional:   the number of components to use in
                                                    each conditional distribution.
        :param nonlinearity: The non-linearity to use at the hidden layer.
        :param name: The name of the specific NADE instance.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_components = number_components_per_conditional
        self.nonlinearity = nonlinearity
        # TODO: Move to sample.
        self.srng = RandomStreams()


        input = Input(shape=(self.input_dim, ))
        pre_activation = NadeInputLayer(self.hidden_dim, self.input_dim, weight_init=weight_init)(input)
        hidden_layer = Activation(self.nonlinearity)(pre_activation)

        output = RNadeOutputLayer(self.input_dim, self.num_components, input_var=input, weight_init=weight_init)(hidden_layer)
        super(RNade, self).__init__(input=input, output=output, name=name)

    @staticmethod
    def cost(x, log_probability_density):
        return -log_probability_density.mean()

    def sample(self, N):

        def sample_recursion(W, V_alpha, V_mu, V_sigma, b_alpha, b_mu, b_sigma, rho, a, x):
            """
            :param W: A (H,) symbolic matrix
            :param V_alpha: A (K, H) symbolic matrix
            :param V_mu: A (K, H) symbolic matrix
            :param V_sigma: A (K, H) symbolic matrix
            :param b_alpha: A (K,) symbolic matrix
            :param b_mu: A (K,) symbolic matrix
            :param b_sigma: A (K,) symbolic matrix
            :param rho: A symbolic scalar
            :param a: An (N, K) symbolic matrix
            :param x: An (N, K) symbolic matrix
            """
            # (N x H)
            a_plus_one = (a + x * W.dimshuffle('x', 0))

            # (N x 1 x H)
            h_plus_one = self.nonlinearity((a_plus_one * rho).dimshuffle(0, 'x', 1))

            # (N x K) <= ((N x 1 x H) (1 x K x H)).sum(2) + (K)
            mus = (h_plus_one * V_mu.dimshuffle('x', 0, 1)).sum(axis=2) + b_mu

            # TODO here we are getting infs when the exponent is too large.
            # (N x K) <= ((N x 1 x H) (1 x K x H)).sum(2) + (K)
            log_sigmas = ((V_sigma * h_plus_one).sum(axis=2) + b_sigma).clip(-20, +20)

            sigmas = T.exp(log_sigmas)

            # TODO here we get NaNs when any element in the exponent is too large giving infs => inf/inf = NaN.
            # TODO and then we get NaNs in the alphas when we go to sample from them.
            # (N x K) <= ((N x 1 x H) (1 x K x H)).sum(2) + (K)
            exponent_alpha = (V_alpha * h_plus_one).sum(axis=2) + b_alpha
            # (N x K) <= (N x K) - (N x 1)
            # One of these at least has to be zero and the rest have to be negative
            exponent_alpha_minus_max = exponent_alpha - exponent_alpha.max(axis=1).dimshuffle(0, 'x')
            # (N x K) <= (N x K)/(N x 1)
            alphas = T.exp(exponent_alpha_minus_max) / T.exp(exponent_alpha_minus_max).sum(1).dimshuffle(0, 'x')

            # (N x K) This returns a matrix of N rows where each row is 1 hot showing which
            # component has been picked.
            c = self.srng.multinomial(pvals=alphas, dtype=theano.config.floatX)
            # (N)
            sigmas_chosen = (sigmas * c).sum(axis=1)
            # (N)
            mus_chosen = (mus * c).sum(axis=1)
            # (N x H)
            x_plus_one = (
            (self.srng.normal(size=(N,)) * sigmas_chosen + mus_chosen).dimshuffle(0, 'x') * np.ones((N, self.hidden_dim),
                                                                                                    dtype=theano.config.floatX))
            return [a_plus_one, x_plus_one]

        a_init = K.zeros((N, self.hidden_dim), dtype="float32")
        x_init = K.ones((N, self.hidden_dim), dtype="float32")

        ((a_vals, x_vals), updates) = theano.scan(fn=sample_recursion,
                                                  sequences=[self.layers[1].W,
                                                             self.layers[-1].V_alpha.dimshuffle(2, 0, 1),
                                                             self.layers[-1].V_mu.dimshuffle(2, 0, 1),
                                                             self.layers[-1].V_sigma.dimshuffle(2, 0, 1),
                                                             self.layers[-1].b_alpha.dimshuffle(1, 0),
                                                             self.layers[-1].b_mu.dimshuffle(1, 0),
                                                             self.layers[-1].b_sigma.dimshuffle(1, 0),
                                                             self.layers[1].rho],
                                                  outputs_info=[a_init, x_init])
        # return x_vals[:, :, 0].T.eval()
        r = x_vals[:, :, 0].T
        f = theano.function(inputs=[], outputs=r, updates=updates)
        return f()


class BinaryNade(Sequential):

    def __init__(self, input_dim, hidden_dim, nonlinearity=K.relu, name="BinaryNade"):
        """
        Initialise a binary valued NADE, as is outlined in the following paper:
        http://homepages.inf.ed.ac.uk/imurray2/pub/11nade/

        :param input_dim: The input dimensionality.
        :param hidden_dim: The hidden layer dimensionality.
        :param nonlinearity: The non-linearity to use at the hidden layer.
        :param name: The name of the specific NADE instance.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity
        super(BinaryNade, self).__init__(name=name)

    def initialise(self, weight_init=glorot_normal):
        """
        Initialise the model. Do this before compiling.
        :param weight_init: The weight initialisation method. Any method in K.initializations
                            can be used.
        """
        self.add(NadeInputLayer(self.hidden_dim, self.input_dim, weight_init=weight_init))
        self.add(Activation(self.nonlinearity))
        self.add(NadeOutputBinaryLayer(self.input_dim, weight_init=weight_init))
        self.add(Activation('sigmoid'))

    @staticmethod
    def cost(x, prob_x):
        """
        Returns the negative log-likelihood of the parameters given the
        input data.

        :param x: (N, D) tensor, where N is the number of examples and D is the
                  dimension of the input data.
        :param prob_x: (N, D) tensor, where the dth dimension in each case is
                       the conditional probability of the dth element of the
                       input being one given all the previous d-1 elements.
        :return: The mean negative log likelihood of the data.
        """
        return -K.mean(K.sum((x * K.log(prob_x) + (1 - x) * K.log(1 - prob_x)), axis=1))

    def sample(self, N):
        """
        Sample N times from the model.
        :param N: The number of samples.
        :return: A (N, input_dim) numpy array.
        """
        def sample_recursion(W, rho, V, b, uniform_random, a, x):
            """
            :param W: A (H,) symbolic matrix.
            :param rho: A scalar.
            :param V: A (H,) symbolic matrix.
            :param b: A symbolic scalar.
            :param uniform_random: A (N, ) symbolic matrix.
            :param a: An (N, ) symbolic matrix.
            :param x: An (N, ) symbolic matrix.
            """
            # (N x H)
            a_plus_one = a + x * W.dimshuffle('x', 0)
            pre_nonlinearity = rho * a_plus_one

            # (N x H)
            h_plus_one = self.nonlinearity(pre_nonlinearity)

            # (N)
            p_x_d = K.sigmoid(T.dot(h_plus_one, V) + b)

            # (N, 1)
            x_plus_one = K.switch(T.gt(p_x_d, uniform_random), 1, 0).dimshuffle(0, "x")
            x_plus_one *= T.ones((1, self.layers[0].output_dim), dtype="float32")

            return a_plus_one, x_plus_one

        a_init = K.zeros((N, self.hidden_dim), dtype="float32")
        x_init = K.ones((N, self.hidden_dim), dtype="float32")

        srng = RandomStreams()
        random_numbers = srng.uniform(size=(self.input_dim, N))
        ((a_vals, x_vals), updates) = theano.scan(fn=sample_recursion,
                                                  sequences=[self.layers[0].W,
                                                             self.layers[0].rho,
                                                             self.layers[2].V.T,
                                                             self.layers[2].b,
                                                             random_numbers],
                                                  outputs_info=[a_init, x_init])

        return x_vals[:, :, 0].T.eval()
