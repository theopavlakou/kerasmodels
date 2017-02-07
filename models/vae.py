'''
This script is a slightly generalised version of
https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py

It implements a Variational Auto-Encoder.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

# Set up all the dimensions and the training parameters.
batch_size = 100
input_dim = 784
latent_dim = 2
intermediate_dim = 256
nb_epoch = 10
epsilon_std = 1.
weights_filename = "weights.h5"
nb_layers_encoder = 3
nb_layers_decoder = 3


def get_input():
    """
    Returns the input to the network encoder.
    :return: Keras Input of dimention input_dim.
    """
    return Input(shape=(input_dim,))


def get_encoder_output_and_loss(input_data, num_hidden_layers=3):
    """

    :param input_data: Keras Input of dimention input_dim.
    :num_hidden_layers: the number of hidden layers in the encoder.
    :return z_mean: the mean of the latent space variables of shape
                        (batch_size, latent_dim).
    :return z_log_var: the log variance of the latent space variables of
                        shape (batch_size, latent_dim).
    :return vae_loss: the loss for the variational autoencoder.
    """
    encoder_hidden_layer = Dense(intermediate_dim,
                                activation='relu')(input_data)
    for i in xrange(1, num_hidden_layers):
        encoder_hidden_layer = Dense(intermediate_dim,
                                    activation='relu')(encoder_hidden_layer)

    z_mean = Dense(latent_dim)(encoder_hidden_layer)
    z_log_var = Dense(latent_dim)(encoder_hidden_layer)

    def vae_loss(output_true, output_mean):
        crossentropy_loss = (input_dim *
                    objectives.binary_crossentropy(output_true, output_mean))
        kl_loss = (-0.5 * K.sum(1 + z_log_var - K.square(z_mean)
                    - K.exp(z_log_var), axis=-1))
        return crossentropy_loss + kl_loss

    return z_mean, z_log_var, vae_loss


def get_latent_variables(z_mean, z_log_var):
    """
    Return the sampling layer in the latent space.
    :param z_mean: the mean of the latent space variables of shape
                    (batch_size, latent_dim).
    :param z_log_var: the log variance of the latent space variables of shape
                    (batch_size, latent_dim).
    :return: the sampling layer in the latent space.
    """
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=z_mean.shape, mean=0., std=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    return Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])


def get_decoder_output_and_model(latent_var, num_hidden_layers=3):
    """
    Get the decoder output and the decoder (generator) network i.e. the part of
    the VAE that maps from the latent space to the data space again.
    :param latent_var: the latent space variables of shape
                        (batch_size, latent_dim).
    :param num_hidden_layers: the number of hidden layers in the decoder.

    :return decoder_output_mean: the output of the decoder of shape
                                    (batch_size, input_dim).
    :return generator:  a Keras model that maps from the latent space to the
                        data space again.
    """
    layers = []
    for l in xrange(num_hidden_layers):
        layers.append(Dense(intermediate_dim, activation='relu'))
    # We will model the output image as binary variables.
    decoder_mean = Dense(input_dim, activation='sigmoid')

    h_decoded = layers[0](latent_var)
    for i in xrange(1, num_hidden_layers):
        h_decoded = layers[i](h_decoded)
    decoder_output_mean = decoder_mean(h_decoded)

    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = layers[0](decoder_input)
    for i in xrange(1, num_hidden_layers):
        _h_decoded = layers[i](_h_decoded)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    return decoder_output_mean, generator


def get_vae_encoder_and_decoder(optimizer='adam', num_hidden_layers_encoder=3,
                                num_hidden_layers_decoder=3):
    """
    Get the VAE, its encoder and its decoder.
    :param optimizer: name of the optimizer to use when optimizing the VAE.
    :num_hidden_layers_encoder: the number of hidden layers in the encoder.
    :num_hidden_layers_decoder: the number of hidden layers in the decoder.

    :return vae:    the VAE which maps from input data to the latent space
                    and back to the input data space.
    :return encoder: a Keras model that maps from the input space to the
                     latent space.
    :return generator:  a Keras model that maps from the latent space to the
                        data space again.
    """
    input_data = get_input()
    z_mean, z_log_var, vae_loss = get_encoder_output_and_loss(input_data,
                                    num_hidden_layers_encoder)
    z = get_latent_variables(z_mean, z_log_var)
    output_mean, generator = get_decoder_output_and_model(z,
                                    num_hidden_layers_decoder)
    encoder = Model(input_data, z_mean)
    vae = Model(input_data, output_mean)
    vae.compile(optimizer=optimizer, loss=vae_loss)

    return vae, encoder, generator

if __name__ == "__main__":
    import os.path as path

    def make_path(relative_path):
        root = path.dirname(path.dirname(__file__))
        return path.join(root, relative_path)

    vae, encoder, generator = get_vae_encoder_and_decoder(
                                num_hidden_layers_encoder=nb_layers_encoder,
                                num_hidden_layers_decoder=nb_layers_decoder)
    # train the VAE on MNIST digits.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    train = True
    weights_path = make_path(weights_filename)

    if train:
        vae.fit(x_train, x_train,
                shuffle=True,
                nb_epoch=nb_epoch,
                batch_size=batch_size,
                validation_data=(x_test, x_test))
        vae.save_weights(weights_path)
    else:
        vae.load_weights(weights_path)

    # figure with 15x15 digits.
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through
    # the inverse CDF (ppf) of the Gaussian to produce values of the latent
    # variables z, since the prior of the latent space is Gaussian.
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
