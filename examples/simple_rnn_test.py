
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.deduce import deduce, maximize, Monitor
from vilab.env import Env
from vilab.datasets import load_mnist_realval

setup_log(logging.DEBUG)

x, y, h = Sequence("x"), Sequence("y"), Sequence("h")
t = Index("t")

f = Function("f")

y[t] == f(x[t], h[t-1])
h[t] == f(y[t])

res = Summation(y[t])


sm = deduce(
	res,
	feed_dict={
		x: np.random.randn(100, 10, 5),
		h[0]: np.zeros((10, 3)),
	},
	structure={
		y: 11
	}
)
