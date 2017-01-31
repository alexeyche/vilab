

from api import *
import tensorflow as tf
import numpy as np
from util import is_sequence

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


class Engine(object):
    CURRENT_SESSION = None

    @classmethod
    def sample(cls, d, shape):
        args = d.get_args()
        if isinstance(d, N):
            N0 = tf.random_normal(shape)
            mu = args[0]
            var = tf.exp(0.5 * args[1])
            if isinstance(mu, tf.Tensor):
                assert mu.get_shape() == var.get_shape(), "Shapes of deduced arguments for {} is not right".format(d)
            return  mu + var * N0
        elif isinstance(d, Point):
            return args[0]
        else:
            raise Exception("Failed to sample {}".format(d))

    @classmethod
    def get_session(cls):
        if cls.CURRENT_SESSION is None:
            cls.open_session()
        return cls.CURRENT_SESSION

    @classmethod
    def open_session(cls):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        cls.CURRENT_SESSION = sess
        return sess


    @classmethod
    def run(cls, *args, **kwargs):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return sess.run(*args, **kwargs)

    @classmethod
    def provide_input(cls, var_name, shape):
        return tf.placeholder(tf.float32, shape=shape, name=var_name)


    @classmethod
    def get_act(cls, act):
        if act == linear:
            return None
        elif act == softplus:
            return tf.nn.softplus
        elif act == relu:
            return tf.nn.relu
        raise Exception("Engine: Can't find activation function: {}".format(act))


    @staticmethod
    def function(*args, **kwargs):
        assert 'size' in kwargs, "Need size information"
        assert 'name' in kwargs, "Need name for output"

        assert len(args) > 0, "Empty args"

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

        if not is_sequence(size):
            size = (size,)
        
        if layers_num is None:
            layers_num = len(size)
        else:
            assert layers_num == len(size), "Got layers num not matched with size information. layers_num: {}, size: {}".format(layers_num, size)
        

        act = None
        if user_act:
            act = Engine.get_act(user_act)

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
                    bias_init = lambda shape, dtype, partition_info: np.zeros((nout,))
                    if not use_weight_norm:
                        w = tf.get_variable("W{}-{}".format(l_id, idx), [nin, nout], dtype = tf.float32, initializer = init)
                        a_w = tf.matmul(a, w)
                    else:
                        V = tf.get_variable("V{}-{}".format(l_id, idx), [nin, nout], dtype = tf.float32, initializer = init) #tf.uniform_unit_scaling_initializer(factor=weight_factor))
                        g = tf.get_variable("g{}-{}".format(l_id, idx), [nout], dtype = tf.float32, initializer = vec_init)

                        a_w = tf.matmul(a, V)
                        a_w = a_w * g/tf.sqrt(tf.reduce_sum(tf.square(V),[0]))

                    if use_bias:
                        b = tf.get_variable("b{}-{}".format(l_id, idx), [nout], tf.float32, initializer = bias_init)
                        a_w = a_w + b

                    layer_out = layer_out + a_w
                inputs = (act(layer_out) if act else layer_out,)

        return inputs[0]


    @classmethod
    def calculate_metrics(cls, metrics):
        if isinstance(metrics, KL):
            pass
        else:
            raise Exception("Met unknown metrics: {}".format(metrics))


