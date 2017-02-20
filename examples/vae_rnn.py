
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.deduce import deduce, maximize, Monitor
from vilab.env import Env
from vilab.datasets import load_mnist_realval

setup_log(logging.DEBUG)

# Task related definitions

h_seq, x_seq, z_seq = Sequence("h"), Sequence("x"), Sequence("z")
t = Index("t")

h_new, h, x, z = h_seq[t], h_seq[t-1], x_seq[t], z_seq[t]    # previous state, current data, current latent data (will be generated)

# Definition of high level functions

p, q = Model("p"), Model("q")

mlp = Function("mlp", act=softplus)

mu, var = Function("mu", mlp), Function("var", mlp)
logit = Function("logit", mlp)
f = Function("f")

# Model definition

p(z) == N(mu(h), var(h))
q(z | h, x) == N(mu(x, h), var(x, h))
p(x | h, z) == B(logit(z))

h_new == f(x, z, h)

# Target value

LL_t = - KL(q(z | h, x), p(z)) + log(p(x | h, z))
LL = Summation(LL_t)

 
x_train = np.random.randn(20, 10)
ndim = x_train.shape[1]

h_o = deduce(
	LL,
	feed_dict={
		x: x_train,
		h: np.zeros((20, 2))
	},
	structure={
		mlp: (200, 200,),
		z: 2,
		h_new: 3
	}
)
