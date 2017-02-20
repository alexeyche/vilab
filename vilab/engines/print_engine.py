
from engine import Engine

class PrintEngine(Engine):
    def __init__(self):
        self._debug_outputs = []

    def sample(self, density, shape, importance_samples):
        return "~{}({})".format(density.get_name(), ", ".join([str(a) for a in density.get_args()]))

    def likelihood(self, density, data):
        return "log({})".format(density)

    def run(self, *args, **kwargs):
        return [", ".join([ str(a) for a in args[0]])]

    def get_optimizer(self, optimizer, learning_rate):
        raise NotImplementedError

    def optimization_output(self, value, optimizer, learning_rate):
        raise NotImplementedError

    def provide_input(self, var_name, shape):
        return var_name

    def get_basic_function(self, bf):
        return str(bf)

    def calc_basic_function(self, bf, *args):
        bf = self.get_basic_function(bf)
        return "{}(\n\t{})".format(bf, "\t, \n".join([str(a) for a in args]))

    def function(self, *args, **kwargs):
        return "{}(\n\t{})".format(kwargs["name"], "\t, \n".join([str(a) for a in args]))

    def calculate_metrics(self, metrics, *args):
        return "{}(\n\t{})".format(metrics.get_name(), "\t, \n".join([str(a) for a in args]))

    def iterate_over_sequence(self, callback):
        raise NotImplementedError

    def get_density(self, density):
        return "{}(\n\t{})".format(density.get_name(), "\t, \n".join([str(a) for a in density.get_args()]))


