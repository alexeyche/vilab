
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.deduce import deduce, maximize, Monitor
from vilab.datasets import load_mnist_realval
from vilab.env import Env

setup_log(logging.DEBUG)


# configuration

Function.configure(
	use_batch_norm = False,
)

N.configure(
	importance_samples = 1
)

# Task related definitions

x, z = Variable("x"), Variable("z")

# Definition of high level functions

p, q = Model("p"), Model("q")

mlp = Function("mlp", act=softplus)

mu, var = Function("mu", mlp), Function("var", mlp)
logit = Function("logit", mlp)

# Model definition

q(z | x) == N(mu(x), var(x))
p(x | z) == B(logit(z))

# Target value

LL = - KL(q(z | x), N0) + log(p(x | z))


x_train, t_train, x_valid, t_valid, x_test, t_test = load_mnist_realval()


ndim = x_train.shape[1]
get_pic = lambda arr: arr.reshape(np.sqrt(ndim), np.sqrt(ndim)).T

env = Env("mnist", clear_pics=True)

def monitor_callback(ep, *args):
	shm(get_pic(args[0][0,:]), file=env.run("x_output_{}.png".format(ep)))
	shs(args[1], file=env.run("z_{}.png".format(ep)), labels=t_test)

out, mon_out = maximize(
	LL, 
	epochs=75,
	learning_rate=0.001,
	feed_dict={x: x_train},
	structure={
		mlp: (200, 200,),
		logit: ndim,
		z: 2
	},
	batch_size=100,
	monitor=Monitor(
		[x, z, KL(q(z | x), N0), log(p(x | z)), mu(x), var(x)],
		freq=5,
		feed_dict={x: x_test},
		callback=monitor_callback
	)
)

shl(
	mon_out[:,2],
	mon_out[:,3]*0.01,
	mon_out[:,4],
	np.exp(0.5*mon_out[:,5]),
	labels = ["KL", "log_p_x", "mu", "var"]
)


m = deduce(
	mu(x), 
	context=log(p(x | z)),
	feed_dict={x: x_test[:5000,:]}, 
	structure={mlp: (200, 200), z: 2, logit: ndim}, 
	reuse=True
)

shs(m, labels=t_test[:5000], file=env.run("embedding.png"))
