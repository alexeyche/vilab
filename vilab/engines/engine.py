
from vilab.api import *
from collections import namedtuple, defaultdict

class BaseEngine(object):
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Engine({})".format(self._name)

    def __call__(self, element, ctx=None):
        raise NotImplementedError

    def get_cached(self, key):
        return None

    def cache(self, key, value):
        pass


class CallbackEngine(object):
    def __init__(self, callback):
        super(CallbackEngine, __init__).format("callback_{}".format(callback.__name__))
        self._callback = callback

    def __call__(self, element, ctx=None):
        return self._callback(element, ctx)



class Engine(BaseEngine):
    Ctx = namedtuple("Ctx", [
        "arguments",
        "structure",
        "scope",
        "density_view",
        "provided_input",
        "input_variable",
    ])

    @staticmethod
    def make_ctx(arguments=None, structure=None, scope="", density_view=Density.View.SAMPLE, provided_input=None, input_variable=None):
        if arguments is None:
            arguments = tuple()
        return Engine.Ctx(arguments, structure, scope, density_view, provided_input, input_variable)


    def __init__(self, name):
        super(Engine, self).__init__(name)

        self._callbacks = {
            Variable: self.variable,
            Function: self.function,
            BasicFunction: self.basic_function,
            Density: self.density,
            IntegralType: self.integral_type,
            Metrics: self.metrics,
        }
        self._verbose = True
        self._cache = {}
        
    def find_callback(self, element):
        callback = None
        
        strong_type_callbacks = [ v for k, v in self._callbacks.iteritems() if type(element) == k]
        inherit_type_callbacks = [ v for k, v in self._callbacks.iteritems() if isinstance(element, k)]
        
        if len(strong_type_callbacks) > 0:
            assert len(strong_type_callbacks) == 1
            callback = strong_type_callbacks[0]
        elif len(inherit_type_callbacks) > 0:
            assert len(inherit_type_callbacks) == 1
            callback = inherit_type_callbacks[0]
        else:
            raise Exception("Can't find callback for element: {}".format(element))
        
        return callback

    def __call__(self, element, ctx=None):
        assert ctx is None or isinstance(ctx, Engine.Ctx), "Engine expecting Engine.Ctx as context"
        if ctx is None:
            ctx = Engine.make_ctx()
        
        if self._verbose:
            logging.debug("{}: met element `{}`".format(self, element))
        callback = self.find_callback(element)

        res = callback(element, ctx)
        
        if self._verbose:
            logging.debug("{}: result {}".format(self, res))
        
        return res


    def variable(self, element, ctx):
        raise NotImplementedError()

    def function(self, element, ctx):
        raise NotImplementedError()

    def basic_function(self, element, ctx):
        raise NotImplementedError()
    
    def density(self, element, ctx):
        raise NotImplementedError()

    def integral_type(self, element, ctx):
        raise NotImplementedError()

    def metrics(self, element, ctx):
        raise NotImplementedError()

    def run(self, elements, feed_dict):
        raise NotImplementedError()

    def optimize(self, to_optimize, optimizer, learning_rate):
        raise NotImplementedError()

    def is_data_engine(self):
        return True

    def get_cached(self, key):
        return self._cache.get(key)

    def cache(self, key, value):
        assert not key in self._cache, "Found key {} already in cache".format(key)
        self._cache[key] = value

