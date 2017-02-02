
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.calc import deduce, maximize
from vilab.datasets import load_iris_dataset
from vilab.env import Env


# setup_log(logging.DEBUG)
setup_log(logging.INFO)

x = Variable("x")
z = Variable("z")

p = Model("p")
q = Model("q")

mlp = Function("mlp", act=elu)

mu = Function("mu", mlp)
var = Function("var", mlp, act=softplus)


q(z | x) == N(mu(x), var(x))
p(x | z) == mu(z)

LL =  - KL(q(z | x), N0) - SquaredLoss(p(x | z), x)

x_v, labs = load_iris_dataset()

x_train, x_valid, x_test = load_mnist_binarized()

env = Env()
env.clear_pics(env.run())


out, mon_out = maximize(
	LL, 
	epochs=1000,
	learning_rate=1e-02,
	feed_dict={x: x_v},
	structure={
		mlp: (10,),
		z: 2
	},
	monitor=[
		z, 
		KL(q(z | x), N0), 
		SquaredLoss(p(x | z), x),
		mu(x), 
		var(x)
	],
	monitor_callback=lambda *args: shs(args[1], labels=labs, file=env.run("z_scatter_{}.png".format(args[0]))),
	monitor_freq=10
)

shl(
	mon_out[:,1],
	mon_out[:,2],
	mon_out[:,3],
	np.exp(0.5 * mon_out[:,4]),
	labels = ["KL", "log_p_x", "mu", "var"]
)

x_sample = deduce(x, feed_dict={z: np.random.randn(99, 2)}, structure={x: x_v.shape[1]}, reuse=True)

shs(x_v, labels=labs)
shs(x_sample)

