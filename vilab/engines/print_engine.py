
from engine import Engine
from vilab.api import *

class PrintEngine(Engine):
    def __init__(self):
        self._debug_outputs = []

    def is_data_engine(self):
        return False

    def sample(self, density, shape, importance_samples):
        return "~{}({})".format(density.get_name(), ", ".join([str(a) for a in density.get_args()]))

    def likelihood(self, density, data):
        return "ll({})".format(density)

    def run(self, *args, **kwargs):
        return [", ".join([ str(a) for a in args[0]])]

    def get_optimizer(self, optimizer, learning_rate):
        raise NotImplementedError

    def optimization_output(self, value, optimizer, learning_rate):
        raise NotImplementedError

    def provide_input(self, var_name, shape):
        return var_name

    def calc_basic_function(self, bf, *args):
        s_bf = self.get_basic_function(bf)
        if bf == Arithmetic.ADD or bf == Arithmetic.SUB or bf == Arithmetic.MUL or bf == Arithmetic.POW:
            return "{} {} {}".format(args[0], s_bf, args[1])
        if bf == Arithmetic.NEG:
            return "{}{}".format(s_bf, args[0])
        return "{}({})".format(s_bf, ", ".join([str(a) for a in args]))


    def get_basic_function(self, bf):
        if bf == linear:
            return None
        elif bf == log:
            return "log"
        elif bf == softplus:
            return "softplus"
        elif bf == relu:
            return "relu"
        elif bf == elu:
            return "elu"
        elif bf == tanh:
            return "tanh"
        elif bf == sigmoid:
            return "sigmoid"
        elif bf == Arithmetic.ADD:
            return "+"
        elif bf == Arithmetic.SUB:
            return "-"
        elif bf == Arithmetic.MUL:
            return "*"
        elif bf == Arithmetic.POW:
            return "^"
        elif bf == Arithmetic.POS:
            return None
        elif bf == Arithmetic.NEG:
            return "-"
        else:
            raise Exception("Unsupported basic function: {}".format(bf))

    def function(self, *args, **kwargs):
        return "{}({})".format(kwargs["name"], ",".join([str(a) for a in args]))

    def calculate_metrics(self, metrics, *args):
        return "{}({})".format(metrics.get_name(), ",".join([str(a) for a in args]))

    def iterate_over_sequence(self, sequence, state, callback, output_size, state_size):
        return "", ""

    def get_density(self, density):
        return "{}({})".format(density.get_name(), ",".join([str(a) for a in density.get_args()]))


