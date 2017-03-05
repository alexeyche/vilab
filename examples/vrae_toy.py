#### TODO, support this
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.deduce import deduce, maximize, Monitor
from vilab.env import Env
from vilab.datasets import load_toy_seq_dataset

from vilab.parser import Parser


setup_log(logging.DEBUG)

def sigmoid(x): return 1.0/(1.0 + np.exp(-x))

# Task related definitions

h, g, x = Sequence("h"), Sequence("g"), Sequence("x")
t = Index("t")

z = Variable("z")

# Definition of high level functions

p, q = Model("p"), Model("q")

mlp = Function("mlp", act=elu)

mu, var = Function("mu", mlp), Function("var", mlp)
logit = Function("logit", mlp)
f = Function("f", mlp)

# Model definition

h[t] == f(x[t], h[t-1])

q(z | h[-1]) == N(mu(h[-1]), var(h[-1]))

g[0] == f(z)
p(x[t] | g[t-1]) == B(logit(g[t-1])) 
g[t] == f(x[t], g[t-1])


# Target value

LL = - KL(q(z | h[-1]), N0) + Summation(log(p(x[t] | g[t-1])))

parser = Parser()
out = parser.deduce(LL)

# x_train, x_labs = load_toy_seq_dataset(batch_size=100, seq_size=25, n_class=5)
# batch_size = x_train.shape[1]

# env = Env("vae_rnn", clear_pics=True)

# def monitor_callback(ep, *args):
# 	shm(sigmoid(args[0][:,0,:]), x_train[:,0,:], file=env.run("x_{}.png".format(ep)))
# 	shl(args[1][:,0,:], file=env.run("z_{}.png".format(ep)))

# out, _, ctx = maximize(
# 	LL, 
# 	epochs=500,
# 	learning_rate=0.1,
# 	feed_dict={
# 		x_seq: x_train,
# 		h0: np.zeros((batch_size, 10))
# 	},
# 	structure={
# 		mlp: (20,),
# 		z: 2
# 	},
# 	monitor=Monitor(
# 		[Iterate(logit(z)), Iterate(mu(x, h)), Summation(KL(q(z | h, x), p(z))), Summation(log(p(x | h, z)))],
# 		freq=5,
# 		callback=monitor_callback
# 	)
# )

# m, _ = deduce(Iterate(mu(x, h)), context=ctx)

# shs(np.mean(m, 0), labels=x_labs)
# shl(*[m[:,bi,:] for bi in xrange(batch_size)])