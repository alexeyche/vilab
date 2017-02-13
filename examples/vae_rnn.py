
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.calc import deduce, maximize, Monitor
from vilab.env import Env


h, x, z = Sequence("h"), Sequence("x"), Sequence("z")
t = Index("t")

p, q = Model("p"), Model("q")

mlp = Function("mlp", act=softplus)

mu = Function("mu", mlp)
var = Function("var", mlp)
logit = Function("logit", mlp)
f = Function("f")


p(z[t]) == N(mu(h[t-1]), var(h[t-1]))
q(z[t] | x[t], h[t-1]) == N(mu(x[t], h[t-1]), var(x[t], h[t-1]))
p(x[t] | z[t]) == B(logit(z[t]))

h[t] == f(x[t], z[t], h[t-1])


LL = - KL( q(z[t] | x[t], h[t-1]), p(z[t])) + log(p(x[t] | z[t]))

