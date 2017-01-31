
import vilab.log

from vilab.api import *
from vilab.calc import sample, deduce
from vilab.util import *
from functools import partial as Partial

x = Variable("x")
z = Variable("z")

p = Model("p")
q = Model("q")

mlp = Function("mlp", act=relu)

mu = Function("mu") | mlp
var = Function("var", act=softplus) | mlp


p(z) == N0()

p(x | z) == N(mu(z), var(z))
q(z | x) == N(mu(x), var(x))

structure = {
	mlp: (100, 50),
	z: 10,
	x: 100
}

LL = - KL(q(z | x), p(z)) + log(p(x | z))

logging.info("==============================================")

engine_inputs = {}

o = deduce(q, LL, set([x, z]), {x: np.random.randn(100, 100)}, structure, 100, engine_inputs)

# y_v = sample(p(x | z), structure=structure, batch_size=100)



# y_v = sample(q(z | x), structure=structure, batch_size=100, feed_dict = {
# 	x: np.random.randn(100, 100)
# })


# maximize(LL, epochs=100, feed_dict = {
# 	x: np.random.randn(100, 100)	
# })
