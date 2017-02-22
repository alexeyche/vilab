
from engine import Engine
from tensor import Tensor
from vilab.api import *
from vilab.util import is_sequence

import numbers

class VarTensor(Tensor):
    def __init__(self, shape, dependencies=[]):
        super(VarTensor, self).__init__(None, shape)
        self._dependencies = dependencies


def get_shape(v):
    if isinstance(v, numbers.Integral) or isinstance(v, numbers.Real):
        return ()
    elif isinstance(v, Tensor):
        return v.get_shape()
    else:
        raise Exception("Unknown element: {}".format(v))

def prom_shapes(shapes):
    assert len(set([len(s) for s in shapes])) == 1
    shape = []
    for dims in zip(*shapes):
        shape.append(max(dims))
    return tuple(shape)

class VarEngine(Engine):
    def __init__(self):
        self._debug_outputs = []

    def is_data_engine(self):
        return False

    def sample(self, density, shape, importance_samples):
        return VarTensor(shape, density.get_args())

    def likelihood(self, density, data):
        return VarTensor(data.get_shape(), density.get_args())

    def run(self, *args, **kwargs):
        return args[0]

    def get_optimizer(self, optimizer, learning_rate):
        raise NotImplementedError

    def optimization_output(self, value, optimizer, learning_rate):
        raise NotImplementedError

    def provide_input(self, var_name, shape):
        return VarTensor(shape)

    def calc_basic_function(self, bf, *args):
        shape = prom_shapes([get_shape(a) for a in args])
        return VarTensor(shape, args) 

    def function(self, *args, **kwargs):
        size = kwargs["size"]
        
        if is_sequence(size):
            size = size[-1]
        
        assert len(args)>0
        assert len(get_shape(args[0])) == 2

        batch_sizes = set([get_shape(a)[0] for a in args])
        assert len(batch_sizes) == 1
        batch_size = batch_sizes.pop()

        return VarTensor((batch_size, size), args)        

    def calculate_metrics(self, metrics, *args):
        assert len(args)>0
        
        batch_sizes = set([get_shape(a)[0] for a in args if len(get_shape(a)) > 0])
        assert len(batch_sizes) == 1
        batch_size = batch_sizes.pop()
        
        return VarTensor((batch_size, 1), args)

    def iterate_over_sequence(self, sequence, state, callback, output_size, state_size):
        raise NotImplementedError

    def get_density(self, density):

        return VarTensor(get_shape(density.get_args()[0]), density.get_args())


