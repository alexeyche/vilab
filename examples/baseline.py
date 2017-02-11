
from vilab.log import setup_log
import logging

import numpy as np
import tensorflow as tf
import time

import matplotlib.pyplot as plt

np.random.seed(1982)
tf.set_random_seed(1982)

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist = read_data_sets('/home/alexeyche/tf/MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples


def load_toy_dataset():
    test_data = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    test_data_classes = [0, 1, 2, 3]
    return test_data, test_data_classes



x_train, labs = load_toy_dataset()


def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.Session()
        self.sess.run(init)
    
    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
            
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1), name="r_h1"),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2), name="r_h2"),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z), name="r_out_mean"),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z), name="r_out_log_sigma")}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32), name="r_b1"),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32), name="r_b2"),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32), name="r_b_out_mean"),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32), name="r_b_out_log_sigma")}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1), name="g_h1"),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2), name="g_h2"),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input), name="g_out_mean"),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input), name="g_out_log_sigma")}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32), name="g_b1"),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32), name="g_b2"),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32), name="g_b_out_mean"),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32), name="g_b_out_log_sigma")}
        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        if self.transfer_fct is tf.nn.softplus:
            print "Found soft+"
            z_log_sigma_sq = \
                tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                       biases['out_log_sigma'])
        else:

            z_log_sigma_sq = tf.nn.softplus(
                tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                       biases['out_log_sigma'])
            )
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                 biases['out_mean']))
        return x_reconstr_mean
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-8 + self.x_reconstr_mean)
                           + (1-self.x) * tf.log(1e-8 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        
        self.kl_loss = tf.reduce_mean(latent_loss)
        self.rec_loss = tf.reduce_mean(reconstr_loss)

        self.cost = tf.reduce_mean(reconstr_loss+latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, kl, rec, cost = self.sess.run((self.optimizer, self.kl_loss, self.rec_loss, self.cost), 
                                  feed_dict={self.x: X})
        return cost, kl, rec
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})


def train_mnist(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    
    import os
    writer = tf.summary.FileWriter("{}/tf_bn".format(os.environ["HOME"]), graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost, kl_cost, rec_cost = 0., 0., 0.

        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost, kc, rc = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost/ n_samples * batch_size
            kl_cost += kc/n_samples * batch_size
            rec_cost += rc/n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("{:.5f} Epoch:".format(time.time()), '%04d' % epoch, 
                  "cost=", "{:.5f}, kl cost={:.5f}, rec cost={:.5f}".format(avg_cost, kl_cost, rec_cost))
    return vae


def train_toy(network_architecture, learning_rate=0.001,
          batch_size=x_train.shape[0], training_epochs=50, display_step=5):
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)

    import os
    writer = tf.summary.FileWriter("{}/tf_bn".format(os.environ["HOME"]), graph=tf.get_default_graph())

    for epoch in range(training_epochs):
        cost, kc, rc = vae.partial_fit(x_train)
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                  "cost=", "{:.5f}, kl cost={:.5f}, rec cost={:.5f}".format(cost, kc, rc))
    return vae


# network_architecture = \
#     dict(n_hidden_recog_1=500, # 1st layer encoder neurons
#          n_hidden_recog_2=500, # 2nd layer encoder neurons
#          n_hidden_gener_1=500, # 1st layer decoder neurons
#          n_hidden_gener_2=500, # 2nd layer decoder neurons
#          n_input=784, # MNIST data input (img shape: 28*28)
#          n_z=20)  # dimensionality of latent space

# vae = train(network_architecture, training_epochs=75)

# x_sample = mnist.test.next_batch(100)[0]
# x_reconstruct = vae.reconstruct(x_sample)

# plt.figure(figsize=(8, 12))
# for i in range(5):

#     plt.subplot(5, 2, 2*i + 1)
#     plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
#     plt.title("Test input")
#     plt.colorbar()
#     plt.subplot(5, 2, 2*i + 2)
#     plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
#     plt.title("Reconstruction")
#     plt.colorbar()
# plt.tight_layout()


network_architecture = \
    dict(n_hidden_recog_1=200, # 1st layer encoder neurons
         n_hidden_recog_2=200, # 2nd layer encoder neurons
         n_hidden_gener_1=200, # 1st layer decoder neurons
         n_hidden_gener_2=200, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=2)  # dimensionality of latent space

vae_2d = train_mnist(network_architecture, training_epochs=75)

x_sample, y_sample = mnist.test.next_batch(5000)
z_mu = vae_2d.transform(x_sample)
plt.figure(figsize=(8, 6)) 
plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
plt.colorbar()
plt.grid()

plt.show()


# from vilab.util import shm

# network_architecture = \
#     dict(n_hidden_recog_1=256, # 1st layer encoder neurons
#          n_hidden_recog_2=256, # 2nd layer encoder neurons
#          n_hidden_gener_1=256, # 1st layer decoder neurons
#          n_hidden_gener_2=256, # 2nd layer decoder neurons
#          n_input=4, # MNIST data input (img shape: 28*28)
#          n_z=2)  # dimensionality of latent space

# vae_2d = train_toy(network_architecture, training_epochs=1000, learning_rate=1e-03)

# z_mu = vae_2d.transform(x_train)
# plt.figure(figsize=(8, 6)) 
# plt.scatter(z_mu[:, 0], z_mu[:, 1], c=labs)
# plt.colorbar()
# plt.grid()

# x_reconstruct = vae_2d.reconstruct(x_train)
# shm(x_reconstruct)
# plt.show()



