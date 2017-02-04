
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.calc import deduce, maximize, Monitor
from vilab.datasets import load_mnist_binarized
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

pb = Function("pb", act=elu)


q(z | x) == N(mu(x), var(x))
p(x | z) == B(pb(z))


LL =  - KL(q(z | x), N0) + log(p(x | z))

x_train, x_valid, x_test = load_mnist_binarized()
ndim = x_train.shape[1]

env = Env()
env.clear_pics(env.run())


out, mon_out = maximize(
	LL, 
	epochs=10,
	learning_rate=3e-04,
	feed_dict={x: x_train},
	structure={
		mlp: (200, 200,),
		pb: (200, 200, ndim),
		z: 100
	},
	batch_size=100,
	monitor=Monitor(
		[KL(q(z | x), N0), log(p(x | z)), mu(x), var(x)],
		freq=1,
		feed_dict={x: x_test}
	)
)

# shl(
# 	mon_out[:,1],
# 	mon_out[:,2],
# 	mon_out[:,3],
# 	np.exp(0.5 * mon_out[:,4]),
# 	labels = ["KL", "log_p_x", "mu", "var"]
# )

# x_sample = deduce(x, feed_dict={z: np.random.randn(100, 2)}, structure={x: ndim}, reuse=True)


