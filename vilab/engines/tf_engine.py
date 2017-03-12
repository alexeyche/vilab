
from vilab.api import *
from vilab.engines.engine import Engine
import tensorflow as tf
import numpy as np
from vilab.util import is_sequence
ds = tf.contrib.distributions
import numbers
from tensorflow.python.ops import rnn_cell as rc
from tensorflow.python.ops import rnn
from collections import OrderedDict

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


def get_optimizer(optimizer, learning_rate):
    if optimizer == Optimizer.ADAM:
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer == Optimizer.SGD:
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        raise Exception("Unsupported optimizer: {}".format(optimizer))


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


def sample(density, args, shape, importance_samples):
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


def get_density(density, args):
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


def calculate_metrics(metrics, args):
    if isinstance(metrics, KL):
        assert len(args) == 2, "Need two arguments for KL metric"
        assert isinstance(args[0], ds.Distribution), "Need argument to KL be distribution object, got {}".format(args[0])
        assert isinstance(args[1], ds.Distribution), "Need argument to KL be distribution object, got {}".format(args[1])

        ret = ds.kl(args[0], args[1])
        ret_sum = tf.reduce_sum(ret, ret.get_shape().ndims-1, keep_dims=True)

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

def get_optimizer(optimizer, learning_rate):
    if optimizer == Optimizer.ADAM:
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer == Optimizer.SGD:
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        raise Exception("Unsupported optimizer: {}".format(optimizer))



class TfEngine(Engine):
    def __init__(self, reuse=False):
        super(TfEngine, self).__init__("TfEngine")
        self._reuse = reuse
        self._inputs = OrderedDict()
        logging.info("Opening TensorFlow session")
        self._session = tf.Session()
        self._initialized = False

    def variable(self, element, ctx):
        if len(ctx.arguments) == 0:
            assert not ctx.structure is None, "Expecting structure for `{}'".format(element)
            if element in self._inputs:
                return self._inputs[element]
            
            inp = tf.placeholder(tf.float32, shape=ctx.structure, name="input_{}".format(element.get_name()))
            self._inputs[element] = inp
            return inp
        else:
            assert len(ctx.arguments) == 1, "Unexpected number of arguments for variable"
            return ctx.arguments[0]

    def function(self, element, ctx):        
        assert not ctx.structure is None, "Expecting structure for `{}'".format(element)
        assert len(ctx.arguments)>0, "Expecting non zero arguments for `{}'".format(element)
        
        cfg = element.get_config()
        # logging.debug("{}: Having Function with structure {} and scope {}".format(self, ctx.structure, ctx.scope))
        return function(
            *ctx.arguments, 
            size = ctx.structure, 
            name = element.get_name(),
            scope_name = ctx.scope,
            act = element.get_act(),
            reuse = self._reuse,
            weight_factor = cfg.weight_factor,
            use_batch_norm = cfg.use_batch_norm if not element.get_act() is None else False
        )

    def basic_function(self, element, ctx):
        assert len(ctx.arguments)>0, "Expecting non zero arguments for `{}'".format(element)
        return get_basic_function(element)(*ctx.arguments)

    def density(self, element, ctx):
        if ctx.density_view == Density.View.SAMPLE:
            return sample(element, ctx.arguments, ctx.structure, 1)
        elif ctx.density_view == Density.View.DENSITY:
            return get_density(element, ctx.arguments)
        elif ctx.density_view == Density.View.PROBABILITY:
            density = get_density(element, ctx.arguments)
            
            data = ctx.provided_input
            if data is None:
                assert not ctx.structure is None, "Need to provide structure for likelihood calculation data"
                assert not ctx.input_variable is None, "Need to provide variable element to make tight with input data"

                if ctx.input_variable in self._inputs:
                    data = self._inputs[ctx.input_variable]
                else:
                    data = tf.placeholder(tf.float32, shape=ctx.structure, name="input_{}".format(element.get_name()))
                    self._inputs[ctx.input_variable] = data


            ret = density._log_prob(data)
        
            ret_sum = tf.reduce_sum(ret, ret.get_shape().ndims-1, keep_dims=True)
            
            return ret_sum
        else:
            raise Exception("Unexpected density view: {}".format(ctx.density_view))


    def integral_type(self, element, ctx):
        return element.get_value()

    def metrics(self, element, ctx):
        return calculate_metrics(element, ctx.arguments)

    def run(self, elements, feed_dict):
        if not self._initialized:
            self._session.run(tf.global_variables_initializer())
            self._initialized = True

        feed_dict_tf = {}
        for k, v in self._inputs.iteritems():
            assert k in feed_dict, "Expecting {} in input feed_dict".format(k)
            feed_dict_tf[v] = feed_dict[k]
        
        return self._session.run(elements, feed_dict=feed_dict_tf)

    def optimize(self, to_optimize, optimizer, learning_rate):
        optimizer_tf = get_optimizer(optimizer, learning_rate)
        
        tvars = tf.trainable_variables()
        grads_raw = tf.gradients(-tf.reduce_mean(to_optimize), tvars)

        grads, _ = tf.clip_by_global_norm(grads_raw, 5.0)
        apply_grads = optimizer_tf.apply_gradients(zip(grads, tvars))

        return apply_grads

