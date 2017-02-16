
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.calc import deduce, maximize, Monitor
from vilab.env import Env

setup_log(logging.INFO)

# Task related definitions

h_seq, x_seq, z_seq = Sequence("h"), Sequence("x"), Sequence("z")
t = Index("t")

h, x, z = h_seq[t-1], x_seq[t], z_seq[t]    # previous state, current data, current latent data (will be generated)

# Definition of high level functions

p, q = Model("p"), Model("q")

mlp = Function("mlp", act=softplus)

mu, var = Function("mu", mlp), Function("var", mlp)
logit = Function("logit", mlp)
f = Function("f")

# Model definition

p(z) == N(mu(h), var(h))
q(z | x, h) == N(mu(x, h), var(x, h))
p(x | z, h) == B(logit(z))

h[t] == f(x, z, h)

# Target value

LL = - KL(q(z | x, h), p(z)) + log(p(x | z))

