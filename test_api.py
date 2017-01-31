
from vilab.log import setup_log
setup_log()

from vilab.api import *
from vilab.calc import sample
from vilab.util import *
from functools import partial as Partial

a = Variable("a")
b = Variable("b")
c = Variable("c")
x = Variable("x")
z = Variable("z")
y = Variable("y")

p = Model("p")
q = Model("q")

mu = Function("mu")
var = Function("var", act=softplus)

f = Function("f", act=relu)
g = Function("g", act=relu)


p(a) == N0()
p(z | a) == N(mu(f(a)), var(f(a)))
p(x | z, a) == g(z, a)


structure = {
	a: 5,
	z: 10,
	x: 100,
	g: (100, 100),
	f: (100, 100),
}

logging.info("==============================================")

y_v = sample(p(x | z, a), structure=structure, batch_size=100)

