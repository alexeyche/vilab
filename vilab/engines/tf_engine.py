
from vilab.api import *
from vilab.engines.engine import Engine
import tensorflow as tf
import numpy as np
from vilab.util import is_sequence
ds = tf.contrib.distributions
import numbers
from tensorflow.python.ops import rnn_cell as rc
from tensorflow.python.ops import rnn

def xavier_init(fan_in, fan_out, const=0.5):
    """Xavier initialization of network weights.

    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow

    :param fan_in: fan in of the network (n_features)
    :param fan_out: fan out of the network (n_components)
    :param const: multiplicative constant
    """
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)

def xavier_vec_init(fan_in, const=1.0):
    low = -const * np.sqrt(6.0 / fan_in)
    high = const * np.sqrt(6.0 / fan_in)
    return tf.random_uniform((fan_in,), minval=low, maxval=high)


def get_basic_function(bf):
    if bf == linear:
        return None
    elif bf == log:
        return tf.log
    elif bf == softplus:
        return tf.nn.softplus
    elif bf == relu:
        return tf.nn.relu
    elif bf == elu:
        return tf.nn.elu
    elif bf == tanh:
        return tf.nn.tanh
    elif bf == sigmoid:
        return tf.nn.sigmoid
    elif bf == Arithmetic.ADD:
        return tf.add
    elif bf == Arithmetic.SUB:
        return tf.sub
    elif bf == Arithmetic.MUL:
        return tf.mul
    elif bf == Arithmetic.POW:
        return tf.pow
    elif bf == Arithmetic.POS:
        return None
    elif bf == Arithmetic.NEG:
        return tf.neg
    else:
        raise Exception("Unsupported basic function: {}".format(bf))


def function(*args, **kwargs):
    assert 'size' in kwargs, "Need size information"
    assert 'name' in kwargs, "Need name for output"

    assert len(args) > 0, "Empty arguments in function {}".format(kwargs["name"])

    size = kwargs["size"]
    name = kwargs["name"]


    config = kwargs.get("config", {})
    for k, v in config.iteritems():
        if not k in kwargs:
            kwargs[k] = v
    user_act = kwargs.get("act")
    use_bias = kwargs.get("use_bias", True)
    weight_factor = kwargs.get("weight_factor", 1.0)
    use_weight_norm = kwargs.get("use_weight_norm", False)
    layers_num = kwargs.get("layers_num")
    reuse = kwargs.get("reuse", False)
    use_batch_norm = kwargs.get("use_batch_norm", False)
    scope_name = kwargs.get("scope_name", "")
    if scope_name:
        name = "{}/{}".format(scope_name, name)
    if use_weight_norm:
        use_bias = False
    
    epsilon = 1e-03

    if not is_sequence(size):
        size = (size,)
    
    if layers_num is None:
        layers_num = len(size)
    else:
        assert layers_num == len(size), "Got layers num not matched with size information. layers_num: {}, size: {}".format(layers_num, size)
    

    act = None
    if user_act:
        act = get_basic_function(user_act)
    
    assert not act is None or use_weight_norm == False, "Can't use batch normalization with linear activation function"

    with tf.variable_scope(name, reuse=reuse) as scope:
        inputs = args

        for l_id in xrange(layers_num):
            nout = size[l_id]
            layer_out = tf.zeros(inputs[0].get_shape().as_list()[:-1] + [nout], dtype=tf.float32)

            for idx, a in enumerate(inputs):
                a_shape = a.get_shape().as_list()

                nin = a_shape[-1]

                init = lambda shape, dtype, partition_info: xavier_init(nin, nout, const = weight_factor)
                vec_init = lambda shape, dtype, partition_info: xavier_vec_init(nout, const = weight_factor)
                zeros_init = lambda shape, dtype, partition_info: np.zeros((nout,))
                ones_init = lambda shape, dtype, partition_info: np.ones((nout,))
                
                if not use_weight_norm:
                    w = tf.get_variable("W{}-{}".format(l_id, idx), [nin, nout], dtype = tf.float32, initializer = init)
                    a_w = tf.matmul(a, w)
                else:
                    V = tf.get_variable("V{}-{}".format(l_id, idx), [nin, nout], dtype = tf.float32, initializer = init) #tf.uniform_unit_scaling_initializer(factor=weight_factor))
                    g = tf.get_variable("g{}-{}".format(l_id, idx), [nout], dtype = tf.float32, initializer = vec_init)

                    a_w = tf.matmul(a, V)
                    a_w = a_w * g/tf.sqrt(tf.reduce_sum(tf.square(V),[0]))

                if use_bias:
                    b = tf.get_variable("b{}-{}".format(l_id, idx), [nout], tf.float32, initializer = zeros_init)
                    a_w = a_w + b


                layer_out = layer_out + a_w
            
            if use_batch_norm:
                batch_mean, batch_var = tf.nn.moments(layer_out, [0])
                layer_out = (layer_out - batch_mean) / tf.sqrt(batch_var + epsilon)

                gamma = tf.get_variable("gamma{}".format(l_id), [nout], dtype = tf.float32, initializer = ones_init)
                beta = tf.get_variable("beta{}".format(l_id), [nout], dtype = tf.float32, initializer = zeros_init)
                
                layer_out = gamma * layer_out + beta

            inputs = (act(layer_out) if act else layer_out,)

    return inputs[0]



class TfEngine(Engine):
    def __init__(self, reuse=False):
        super(TfEngine, self).__init__("TfEngine")
        self._reuse = reuse

    def variable(self, element, *args):
        if len(args) == 0:
            assert element.has_structure(), "Need to provide structure for variable: {}".format(element)
            inp = tf.placeholder(tf.float32, shape=element.get_structure(), name="input_{}".format(element.get_name()))
            return inp

        raise Exception()

    def function(self, element, *args):
        assert element.has_structure()
        
        cfg = element.get_config()
        
        return function(
            *args, 
            size = element.get_structure(), 
            name = element.get_name(),
            scope_name = element.get_name(),
            act = element.get_act(),
            reuse = self._reuse,
            weight_factor = cfg.weight_factor,
            use_batch_norm = cfg.use_batch_norm if not element.get_act() is None else False
        )

    def basic_function(self, element, *args):
        return get_basic_function(element)(*args)
