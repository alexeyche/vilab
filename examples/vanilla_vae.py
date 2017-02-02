
import logging

from vilab.log import setup_log
from vilab.api import *
from vilab.util import *
from vilab.calc import deduce, maximize
from vilab.datasets import load_iris_dataset
from vilab.env import Env


# setup_log(logging.DEBUG)
setup_log(logging.INFO)

# root = logging.getLogger()
    
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

env = Env()
env.clear_pics(env.run())


def plot_interm(epoch, kl, log_px, z_v, x_res_v):
    if z_v.shape[1] > 2:
        import sklearn.decomposition as dec
        pca = dec.PCA(2)
        z_v = pca.fit(z_v).transform(z_v)

    plt.scatter(z_v[:,0], z_v[:,1], c=labs)
    plt.savefig(env.run("z_scatter{}.png".format(epoch)))
    plt.clf()
    logging.info("\tSquared loss: {}".format(np.mean(np.square(x_res_v - x_v))))


out, mon_out = maximize(
	LL, 
	epochs=2000, 
	feed_dict={x: x_v},
	structure={
		mlp: (10, 5),
		z: 2,
		x: 4
	},
	config={"learning_rate": 1e-03},
	monitor=[KL(q(z | x), N0), log(p(x | z)), z, x],
	monitor_callback=plot_interm,
	monitor_freq=100
)
