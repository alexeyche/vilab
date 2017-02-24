

from vilab.api import *
import tensorflow as tf
import numpy as np
from vilab.util import is_sequence
ds = tf.contrib.distributions
import numbers
from tensorflow.python.ops import rnn_cell as rc
from tensorflow.python.ops import rnn

from engine import Engine

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


class TfEngine(Engine):
    
    def __init__(self):
        super(TfEngine, self).__init__("Tf")
        self.CURRENT_SESSION = None
        self.INITIALIZED = False
    
        self._debug_input_data = {}
        self._debug_outputs = []

    def sample(self, density, shape, importance_samples):
        args = density.get_args()
        
        # shape = (shape[0]*importance_samples,) + shape[1:]    
        
        if isinstance(density, N):
            assert len(shape) == 2, "Unexpected shape"

            N0 = tf.random_normal((shape[0], importance_samples, shape[1]))

            mu = args[0]
            stddev = tf.exp(0.5 * args[1])
            
            mu = tf.reshape(mu, (shape[0], 1, shape[1]))
            stddev = tf.reshape(stddev, (shape[0], 1, shape[1]))
            
            if isinstance(mu, tf.Tensor):
                assert mu.get_shape() == stddev.get_shape(), "Shapes of deduced arguments for {} is not right".format(density)
            
            res = tf.add(mu, tf.mul(stddev, N0), name="sample_{}".format(density.get_name()))
            # return res
            return tf.reshape(
                res, 
                (shape[0]*importance_samples, shape[1]), 
                name="sample_{}".format(density.get_name())
            )
        elif isinstance(density, B):
            U0 = tf.random_uniform(shape)
            sample = tf.less(U0, tf.nn.sigmoid(args[0]), name="sample_{}".format(density.get_name()))
            return tf.cast(sample, args[0].dtype)
        elif isinstance(density, DiracDelta):
            return args[0]
        else:
            raise Exception("Failed to sample {}".format(density))

    
    def get_density(self, density):
        args = density.get_args()
        if isinstance(density, N):
            mu = args[0]
            stddev = tf.exp(0.5 * args[1])
            
            if isinstance(mu, tf.Tensor):
                assert mu.get_shape() == stddev.get_shape(), "Shapes of deduced arguments for {} is not right".format(density)

            # mu = tf.Print(mu,[mu], "mu", summarize=5)
            # mu = tf.Print(mu,[stddev], "sigma", summarize=5)

            return ds.Normal(mu, stddev, name=density.get_name())
        elif isinstance(density, B):
            logits = args[0]
            return ds.Bernoulli(logits=logits)
        elif isinstance(density, DiracDelta):
            return DiracDeltaDistribution(args[0])
        else:
            raise Exception("Failed to get density object: {}".format(density))

    
    def get_shape(self, elem):
        if isinstance(elem, tf.Tensor):
            return elem.get_shape().as_list()
        elif isinstance(elem, DiracDeltaDistribution):
            return elem.point.get_shape().as_list()
        elif isinstance(elem, ds.Normal):
            return elem.mu.get_shape().as_list()
        elif isinstance(elem, ds.Bernoulli):
            return elem.logits.get_shape().as_list()
        elif isinstance(elem, numbers.Integral) or isinstance(elem, numbers.Real):
            return []
        raise Exception("Unknown type: {}".format(elem))

    
    def likelihood(self, density, data):
        density_obj = self.get_density(density)
        
        ret = density_obj._log_prob(data)
    
        ret_sum = tf.reduce_sum(ret, ret.get_shape().ndims-1, keep_dims=True)
        
        return ret_sum
        
    def get_session(self):
        if self.CURRENT_SESSION is None:
            self.open_session()
        return self.CURRENT_SESSION

    def open_session(self):
        sess = tf.Session()
        self.CURRENT_SESSION = sess
        return sess


    def run(self, *args, **kwargs):
        sess = self.get_session()
        if not self.INITIALIZED:
            sess.run(tf.global_variables_initializer())
            self.INITIALIZED = True

        return sess.run(*args, **kwargs)

    
    def get_optimizer(self, optimizer, learning_rate):
        if optimizer == Optimizer.ADAM:
            return tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == Optimizer.SGD:
            return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            raise Exception("Unsupported optimizer: {}".format(optimizer))

    
    def optimization_output(self, value, optimizer, learning_rate):
        optimizer_tf = self.get_optimizer(optimizer, learning_rate)
        
        tvars = tf.trainable_variables()
        grads_raw = tf.gradients(-tf.reduce_mean(value), tvars)

        grads, _ = tf.clip_by_global_norm(grads_raw, 5.0)
        apply_grads = optimizer_tf.apply_gradients(zip(grads, tvars))

        return apply_grads

    
    def provide_input(self, var_name, shape):
        inp = tf.placeholder(tf.float32, shape=shape, name="input_{}".format(var_name))
        self._debug_input_data[var_name] = inp
        return inp

    def get_basic_function(self, bf):
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
    
    def function(self, *args, **kwargs):
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
            act = self.get_basic_function(user_act)
        
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



    
    def calculate_metrics(self, metrics, *args):
        if isinstance(metrics, KL):
            assert len(args) == 2, "Need two arguments for KL metric"
            assert isinstance(args[0], ds.Distribution), "Need argument to KL be distribution object, got {}".format(args[0])
            assert isinstance(args[1], ds.Distribution), "Need argument to KL be distribution object, got {}".format(args[1])

            
            ret = ds.kl(args[0], args[1])

            # ret = -0.5 * tf.reduce_sum(1 + tf.log(args[0].sigma)
            #                                    - tf.square(args[0].mu) 
            #                                    - args[0].sigma, 1)
            # ret = tf.Print(ret,[ret], "ret", summarize=20)
            
            # ret = tf.Print(ret,[args[0].mu], "mu", summarize=20)
            # ret = tf.Print(ret,[args[0].sigma], "sigma", summarize=20)

            # ret = tf.Print(ret,[args[1].mu], "mu0", summarize=20)
            # ret = tf.Print(ret,[args[1].sigma], "sigma0", summarize=20)

            # ret = 0.5*(1 + tf.log(args[0].sigma ** 2) - args[0].mu**2 - args[0].sigma ** 2)
            # return ret
            ret_sum = tf.reduce_sum(ret, ret.get_shape().ndims-1, keep_dims=True)
            
            # from datasets import load_mnist_binarized_small
            # from util import shm

            # sess = self.get_session()

            # ret_v, f_mu, f_var, s_mu, s_var, ret_sum_v = sess.run([ret, args[0].mu, args[0].sigma, args[1].mu, args[1].sigma, ret_sum], 
            #     feed_dict={self._debug_input_data["x"]: load_mnist_binarized_small()[0][:100]})
            

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


    
    def iterate_over_sequence(self, sequence, state, callback, output_size, state_size):
        cell = ArbitraryRNNCell(callback, output_size, state_size)
        out_gen, finstate_gen = rnn.dynamic_rnn(
            cell, 
            sequence, 
            initial_state=state,
            time_major=True
        )
        return out_gen, finstate_gen



class ArbitraryRNNCell(rc.RNNCell):
    def __init__(self, calc_callback, output_size, state_size):
        self.calc_callback = calc_callback
        self._output_size = output_size
        self._state_size = state_size
        
    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def __call__(self, input_tuple, state_tuple, scope=None):
        out, state = self.calc_callback(input_tuple, state_tuple)
        print out, state
        return out, state


