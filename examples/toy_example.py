
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.calc import deduce, maximize, Monitor
from vilab.datasets import load_toy_dataset
from vilab.env import Env

setup_log(logging.INFO)

x, z = Variable("x"), Variable("z")
p, q = Model("p"), Model("q")

mlp = Function("mlp", act=softplus)

mu, var = Function("mu", mlp), Function("var", mlp)
logit = Function("logit", mlp)


q(z | x) == N(mu(x), var(x))
p(x | z) == B(logit(z))


LL = - KL(q(z | x), N0) + log(p(x | z))

x_train, x_classes = load_toy_dataset()

batch_size, ndim = x_train.shape


env = Env("toy_example", clear_pics=True)

structure = {
	mlp: (256,256),
	z: 2,
	logit: ndim
}

def monitor_callback(ep, *args):
	shm(args[0], file=env.run("x_output_{}.png".format(ep)))
	shs(args[1], file=env.run("z_output_{}.png".format(ep)), labels=x_classes)

	
out, mon_out = maximize(
	LL, 
	epochs=1000,
	learning_rate=1e-03,
	optimizer=Optimizer.ADAM,
	feed_dict={x: x_train},
	structure=structure,
	batch_size=batch_size,
	monitor=Monitor(
		[x, z, KL(q(z | x), N0), log(p(x | z)), mu(x), var(x)],
		freq=100,
		callback=monitor_callback
	)
)


x_logit = deduce(
	logit(z), 
	context=log(p(x | z)),
	feed_dict={x: x_train}, 
	structure=structure, 
	reuse=True
)

z_mu_embed = deduce(
	mu(x), 
	context=log(p(x | z)),
	feed_dict={x: x_train}, 
	structure=structure, 
	reuse=True
)

shl(
	mon_out[:,2],
	mon_out[:,3],
	mon_out[:,4],
	np.exp(0.5 * mon_out[:,5]),
	labels = ["KL", "log_p_x", "mu", "var"],
	file = env.run("result.png")
)

shm(np.clip(x_logit, 0.0, 1.0))
shs(z_mu_embed, labels=x_classes)
