
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.calc import deduce, maximize, Monitor
from vilab.datasets import load_toy_dataset
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

x_train, x_classes = load_toy_dataset()
batch_size, ndim = x_train.shape

env = Env("toy_example", clear_pics=True)

structure = {
	mlp: (20,),
	z: 2,
	logit: (20, ndim)
}

def monitor_callback(ep, *args):
	x_sample = deduce(x, feed_dict={z: np.random.randn(batch_size, 2)}, structure={x: ndim}, reuse=True, silent=True)
	shm(x_sample, file=env.run("x_sample_{}.png".format(ep)))
	shm(args[0], file=env.run("x_output_{}.png".format(ep)))
	shs(args[1], file=env.run("z_output_{}.png".format(ep)), labels=x_classes)

	# x_logit = deduce(
	# 	logit(z), 
	# 	model=p,
	# 	feed_dict={x: x_train}, 
	# 	structure=structure, 
	# 	reuse=True
	# )
	# shm(x_logit, file=env.run("x_logit_{}.png".format(ep)))


# out, mon_out = maximize(
# 	LL, 
# 	epochs=150,
# 	learning_rate=1e-02,
# 	feed_dict={x: x_train},
# 	structure=structure,
# 	batch_size=batch_size,
# 	monitor=Monitor(
# 		[x, z, KL(q(z | x), N0), log(p(x | z)), mu(x), var(x)],
# 		freq=10,
# 		callback=monitor_callback
# 	)
# )

# shl(
# 	mon_out[:,2],
# 	mon_out[:,3],
# 	mon_out[:,4],
# 	np.exp(0.5 * mon_out[:,5]),
# 	labels = ["KL", "log_p_x", "mu", "var"],
# 	file = env.result("result.png")
# )

# x_logit = deduce(
# 	logit(z), 
# 	model=p,
# 	feed_dict={x: x_train}, 
# 	structure=structure, 
# 	reuse=True
# )


# x_v = deduce(
# 	x, 
# 	feed_dict={x: x_train}, 
# 	structure=structure, 
# 	reuse=True
# )

