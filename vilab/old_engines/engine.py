

class Engine(object):
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Engine({})".format(self._name)

    def is_data_engine(self):
        return True

    def sample(self, density, shape, importance_samples):
        raise NotImplementedError

    def get_density(self, density):
        raise NotImplementedError

    def get_shape(self, elem):
        raise NotImplementedError

    def likelihood(self, density, data):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def get_optimizer(self, optimizer, learning_rate):
        raise NotImplementedError

    def optimization_output(self, value, optimizer, learning_rate):
        raise NotImplementedError

    def provide_input(self, var_name, shape):
        raise NotImplementedError

    def get_basic_function(self, bf):
        raise NotImplementedError

    def calc_basic_function(self, bf, *args):
        deduced_bf = self.get_basic_function(bf)
        if deduced_bf is None:
            assert len(args) == 1, "Calling empty basic function {} with list of arguments: {}".format(bf, args)
            return args[0]
        return deduced_bf(*args)

    def function(self, *args, **kwargs):
        raise NotImplementedError

    def calculate_metrics(self, metrics, *args):
        raise NotImplementedError        

    def iterate_over_sequence(self, sequence, state, callback, output_size, state_size):
        raise NotImplementedError

    def sequence_operation(self, op, seq):
        raise NotImplementedError        