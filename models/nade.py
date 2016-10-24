from __future__ import division
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.initializations import glorot_normal
from keras.layers import Activation
from keras import backend as K
from theano import tensor as T
import theano
from theano.tensor.shared_randomstreams import RandomStreams


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

    def call(self, x, mask=None):
        return (self.V * x).sum(axis=1) + self.b

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim


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
