
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.deduce import deduce, maximize, Monitor
from vilab.env import Env
from vilab.datasets import load_mnist_realval
from vilab.engines.print_engine import PrintEngine

from vilab.parser import Parser


setup_log(logging.DEBUG)

x, y, h = Sequence("x"), Sequence("y"), Sequence("h")
t = Index("t")

Function.configure(
	weight_factor = 0.1
)

f = Function("f")

y[t] == f(x[t], h[t-1])
h[t] == f(y[t])

cost = - Summation(SquaredLoss(y[t], x[t]))

###########

parser = Parser()
out = parser.deduce(cost)


# x_train = 0.1*np.random.randn(100, 10, 5)



# def monitor_callback(ep, *args):
# 	logging.info("\tloss: {}".format(np.mean(np.square(args[0] - x_train))))

# out, _, ctx = maximize(
# 	cost, 
# 	epochs=75,
# 	learning_rate=0.1,
# 	feed_dict={
# 		x: x_train,
# 		h[0]: np.zeros((10, 3))
# 	},
# 	structure={
# 		y: 5
# 	},
# 	monitor=Monitor(
# 		[y, h],
# 		freq=5,
# 		callback=monitor_callback
# 	)
# )
