
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.calc import deduce, maximize, Monitor
from vilab.datasets import load_mnist_realval
from vilab.env import Env

setup_log(logging.INFO)

x, z = Variable("x"), Variable("z")
p, q = Model("p"), Model("q")


mlp = Function("mlp", act=softplus)

mu = Function("mu", mlp)
var = Function("var", mlp)
logit = Function("logit", mlp)


q(z | x) == N(mu(x), var(x))
p(x | z) == B(logit(z))


LL = - KL(q(z | x), N0) + p(x | z)

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
		[x, z, KL(q(z | x), N0), p(x | z)],
		freq=5,
		feed_dict={x: x_test[:100]},
		# callback=monitor_callback
	)
)

shl(
	mon_out[:,2],
	mon_out[:,3]*0.01,
	labels = ["KL", "log_p_x"]
)


m = deduce(
	mu(x), 
	model=q,
	feed_dict={x: x_test[:5000,:]}, 
	structure={mlp: (200, 200), mu: 2}, 
	reuse=True
)

shs(m, labels=t_test[:5000], file=env.run("embedding.png"))
