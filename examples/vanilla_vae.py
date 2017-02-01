
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.calc import deduce, maximize
from vilab.datasets import load_iris_dataset



setup_log(logging.DEBUG)
# setup_log(logging.INFO)


x = Variable("x")
z = Variable("z")

p = Model("p")
q = Model("q")

mlp = Function("mlp", act=relu)

mu = Function("mu", mlp)
var = Function("var", mlp, act=softplus)


q(z | x) == N(mu(x), var(x))
p(x | z) == N(mu(z), var(z))

LL = - KL(q(z | x), N0) + log(p(x | z))

logging.info("==============================================")

x_v, labs = load_iris_dataset()
batch_size = x_v.shape[0]


o = deduce(
	LL,
	feed_dict={x: x_v},
	structure={
		mlp: (10, 5),
		z: 2,
		x: 4
	}
)

# out = maximize(
# 	LL, 
# 	epochs=200, 
# 	feed_dict={x: x_v},
# 	structure={
# 		mlp: (10, 5),
# 		z: 2,
# 		x: 4
# 	},
# 	config = {"learning_rate": 1e-02},
# 	monitor = [log(p(x | z))]
# )
