
import logging
from random import shuffle

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.deduce import deduce, maximize, Monitor
from vilab.env import Env
from vilab.datasets import load_mnist_realval

setup_log(logging.INFO)

def sigmoid(x): return 1.0/(1.0 + np.exp(-x))

def generate_test_seq(seq_size=25, n_class=5):
    batch_size = 100

    # x_class_a = np.zeros((10, 10))
    # for i in xrange(10):
    #     x_class_a[i, i] = 1.0

    # x_class_b = np.zeros((10, 10))
    # for i in xrange(10):
    #     x_class_b[9-i, i] = 1.0
    

    classes, labs = [], []
    for i in xrange(n_class):
        x_class = np.zeros((seq_size, 10))
        x_class[np.where(0.1 > np.random.random((seq_size, 10)))] = 1.0
        classes += [x_class]*(batch_size/n_class)
        labs += [str(i)]*(batch_size/n_class)

    ids = range(0, len(classes))
    shuffle(ids)
    
    classes = [classes[id] for id in ids]
    labs = [labs[id] for id in ids]

    x_train = np.dstack(classes)
    x_train = np.transpose(x_train, (0,2,1))

    return x_train, labs



# Task related definitions

h_seq, x_seq, z_seq = Sequence("h"), Sequence("x"), Sequence("z")
t = Index("t")

Function.configure(
	weight_factor = 0.2
)


h0, h, h_new = h_seq[0], h_seq[t-1], h_seq[t]   # init state, previous state, new state
x, z = x_seq[t], z_seq[t]  # current data, current latent data (will be generated)

# Definition of high level functions

p, q = Model("p"), Model("q")

mlp = Function("mlp", act=elu)

mu, var = Function("mu", mlp), Function("var", mlp)
logit = Function("logit", mlp)
f = Function("f", mlp)

# Model definition

p(z) == N(mu(h), var(h))
q(z | h, x) == N(mu(x, h), var(x, h))
p(x | h, z) == B(logit(z))

h_new == f(x, h)

# Target value

LL_t = - KL(q(z | h, x), p(z)) + log(p(x | h, z))
# LL_t = log(p(x | h, z))
LL = Summation(LL_t)


x_train, x_labs = generate_test_seq()

T = x_train.shape[0]
batch_size = x_train.shape[1]

env = Env("vae_rnn", clear_pics=True)

def monitor_callback(ep, *args):
	shm(sigmoid(args[0][:,0,:]), x_train[:,0,:], file=env.run("x_{}.png".format(ep)))
	shl(args[1][:,0,:], file=env.run("z_{}.png".format(ep)))

out, _, ctx = maximize(
	LL, 
	epochs=500,
	learning_rate=0.1,
	feed_dict={
		x_seq: x_train,
		h0: np.zeros((batch_size, 10))
	},
	structure={
		mlp: (20,),
		z: 2
	},
	monitor=Monitor(
		[Iterate(logit(z)), Iterate(mu(x, h)), Summation(KL(q(z | h, x), p(z))), Summation(log(p(x | h, z)))],
		freq=5,
		callback=monitor_callback
	)
)

m, _ = deduce(Iterate(mu(x, h)), context=ctx)

shs(np.mean(m, 0), labels=x_labs)

# shs(
#  	m.reshape(m.shape[0]*m.shape[1], 2), 
#  	labels=[ str(subel) for el in [ [l]*T for l in x_labs ] for subel in el ]
# )

shl(*[m[:,bi,:] for bi in xrange(batch_size)])