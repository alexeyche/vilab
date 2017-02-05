
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.calc import deduce, maximize, Monitor
from vilab.datasets import load_mnist_binarized, load_mnist_binarized_small
from vilab.env import Env

# setup_log(logging.DEBUG)
setup_log(logging.INFO)

x, z = Variable("x"), Variable("z")
p, q = Model("p"), Model("q")


mlp = Function("mlp", act=elu)

mu = Function("mu", mlp)
var = Function("var", mlp, act=softplus)

logit = Function("logit", act=elu)


q(z | x) == N(mu(x), var(x))
p(x | z) == B(logit(z))


LL = - KL(q(z | x), N0) + log(p(x | z))

x_train, x_valid, x_test = load_mnist_binarized()
ndim = x_train.shape[1]
get_pic = lambda arr: arr.reshape(np.sqrt(ndim), np.sqrt(ndim)).T

env = Env("mnist", clear_pics=True)

def monitor_callback(ep, *args):
	x_sample = deduce(x, feed_dict={z: np.random.randn(100, 100)}, structure={x: ndim}, reuse=True, silent=True)
	shm(get_pic(x_sample[0,:]), file=env.run("x_sample_{}.png".format(ep)))
	shm(get_pic(args[0][0,:]), file=env.run("x_output_{}.png".format(ep)))
	

out, mon_out = maximize(
	LL, 
	epochs=1000,
	learning_rate=3e-04,
	feed_dict={x: x_train},
	structure={
		mlp: (200, 200,),
		logit: (200, 200, ndim),
		z: 100
	},
	batch_size=100,
	monitor=Monitor(
		[x, KL(q(z | x), N0), log(p(x | z)), mu(x), var(x)],
		freq=1,
		feed_dict={x: x_test},
		callback=monitor_callback
	)
)

shl(
	mon_out[:,0],
	mon_out[:,1],
	mon_out[:,2],
	np.exp(0.5 * mon_out[:,3]),
	labels = ["KL", "log_p_x", "mu", "var"]
)
