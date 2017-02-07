

from api import *
import tensorflow as tf
import numpy as np
from util import is_sequence
ds = tf.contrib.distributions

class DiracDeltaDistribution(ds.Distribution):
    def __init__(self, point, name="DiracDelta"):
        self.point = point

    def _point(self):
        return self.point


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
    INITIALIZED = False
    
    _debug_input_data = {}
    _debug_outputs = []

    @classmethod
    def sample(cls, density, shape):
        args = density.get_args()
        if isinstance(density, N):
            N0 = tf.random_normal(shape)
            mu = args[0]
            stddev = tf.exp(0.5 * args[1])
            if isinstance(mu, tf.Tensor):
                assert mu.get_shape() == stddev.get_shape(), "Shapes of deduced arguments for {} is not right".format(density)
            return tf.add(mu, stddev * N0, name="sample_{}".format(density.get_name()))
        elif isinstance(density, B):
            U0 = tf.random_uniform(shape)
            sample = tf.less(U0, tf.nn.sigmoid(args[0]), name="sample_{}".format(density.get_name()))
            return tf.cast(sample, args[0].dtype)
        elif isinstance(density, DiracDelta):
            return args[0]
        else:
            raise Exception("Failed to sample {}".format(density))


    @classmethod
    def get_density(cls, density):
        args = density.get_args()
        if isinstance(density, N):
            mu = args[0]
            stddev = tf.exp(0.5 * args[1])
            if isinstance(mu, tf.Tensor):
                assert mu.get_shape() == stddev.get_shape(), "Shapes of deduced arguments for {} is not right".format(density)
            return ds.Normal(mu, stddev, name=density.get_name())
        elif isinstance(density, B):
            logits = args[0]
            return ds.Bernoulli(logits=logits)
        elif isinstance(density, DiracDelta):
            return DiracDeltaDistribution(args[0])
        else:
            raise Exception("Failed to get density object: {}".format(density))


    @classmethod
    def likelihood(cls, density, data, logform=False):
        density_obj = cls.get_density(density)
        
        # clipped_data = tf.where(data >= 1e-08, data, 1e-08 * tf.ones_like(data))
        
        ret = density_obj._log_prob(data)
        
        # from datasets import load_mnist_binarized_small
        # from util import shm

        # sess = cls.get_session()
        # ret_v, data_v = sess.run([ret, data], 
        #     feed_dict={
        #         cls._debug_input_data["x"]: load_mnist_binarized_small()[0][:100],
        #         cls._debug_input_data["Binomial"]: load_mnist_binarized_small()[0][:100]
        #     })
        

        # shm(ret_v, data_v)


        ret_sum = tf.reduce_mean(ret, ret.get_shape().ndims-1, keep_dims=True)
        
        if logform:
            return ret_sum
        else:
            return tf.exp(ret_sum)
        
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
        sess = cls.get_session()
        if not cls.INITIALIZED:
            sess.run(tf.global_variables_initializer())
            cls.INITIALIZED = True

        return sess.run(*args, **kwargs)


    @classmethod
    def get_optimizer(cls, optimizer, learning_rate):
        if optimizer == Optimizer.ADAM:
            return tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == Optimizer.SGD:
            return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            raise Exception("Unsupported optimizer: {}".format(optimizer))

    @classmethod
    def optimization_output(cls, value, optimizer, learning_rate):
        optimizer_tf = cls.get_optimizer(optimizer, learning_rate)
        return optimizer_tf.minimize(-tf.reduce_mean(value))

    @classmethod
    def provide_input(cls, var_name, shape):
        inp = tf.placeholder(tf.float32, shape=shape, name="input_{}".format(var_name))
        cls._debug_input_data[var_name] = inp
        return inp


    @classmethod
    def get_basic_function(cls, bf):
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

    @classmethod
    def calc_basic_function(cls, bf, *args):
        deduced_bf = cls.get_basic_function(bf)
        if deduced_bf is None:
            assert len(args) == 1, "Calling empty basic function {} with list of arguments: {}".format(bf, args)
            return args[0]
        return deduced_bf(*args)

    @staticmethod
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

        if not is_sequence(size):
            size = (size,)
        
        if layers_num is None:
            layers_num = len(size)
        else:
            assert layers_num == len(size), "Got layers num not matched with size information. layers_num: {}, size: {}".format(layers_num, size)
        

        act = None
        if user_act:
            act = Engine.get_basic_function(user_act)

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
    def calculate_metrics(cls, metrics, *args):
        if isinstance(metrics, KL):
            assert len(args) == 2, "Need two arguments for KL metric"
            assert isinstance(args[0], ds.Distribution), "Need argument to KL be distribution object, got {}".format(args[0])
            assert isinstance(args[1], ds.Distribution), "Need argument to KL be distribution object, got {}".format(args[1])

            ret = ds.kl(args[0], args[1])
            # ret = -0.5*(1 + tf.log(args[0].sigma ** 2) - args[0].mu**2 - args[0].sigma ** 2)

            ret_sum = tf.reduce_sum(ret, ret.get_shape().ndims-1, keep_dims=True)
            
            # from datasets import load_mnist_binarized_small
            # from util import shm

            # sess = cls.get_session()

            # ret_v, f_mu, f_var, s_mu, s_var, ret_sum_v = sess.run([ret, args[0].mu, args[0].sigma, args[1].mu, args[1].sigma, ret_sum], 
            #     feed_dict={cls._debug_input_data["x"]: load_mnist_binarized_small()[0][:100]})
            

            # shm(ret_v, f_mu, f_var)

            return ret_sum
        elif isinstance(metrics, SquaredLoss):
            assert len(args) == 2, "Need two arguments for SquaredLoss metric"
            
            assert isinstance(args[0], DiracDeltaDistribution) or isinstance(args[0], tf.Tensor), \
                "Need argument to SquaredLoss be dirac delta distribution object, got {}".format(args[0])
            assert isinstance(args[1], DiracDeltaDistribution) or isinstance(args[1], tf.Tensor), \
                "Need argument to SquaredLoss be dirac delta distribution object, got {}".format(args[1])


            left, right = args[0], args[1]
            if isinstance(left, DiracDeltaDistribution):
                left = left._point()
            if isinstance(right, DiracDeltaDistribution):
                right = right._point()
            
            ret = tf.square(left - right)

            return tf.reduce_sum(ret, ret.get_shape().ndims-1, keep_dims=True)/2.0
        else:
            raise Exception("Met unknown metrics: {}".format(metrics))


