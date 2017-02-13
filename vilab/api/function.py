
import logging
from ..util import is_sequence
from ..config import Config

class BasicFunction(object):
    def __init__(self, name):
        self._name = name

    def __call__(self, *args):
        return FunctionResult(self, *args)

    def __str__(self):
        return "BasicFunction({})".format(self.get_name())

    def __repr__(self):
        return str(self)

    def get_name(self):
        return self._name


class Arithmetic(object):
    ADD = BasicFunction("add")
    SUB = BasicFunction("sub")
    MUL = BasicFunction("mul")
    DIV = BasicFunction("div")
    POW = BasicFunction("pow")
    POS = BasicFunction("pos")
    NEG = BasicFunction("neg")

    def __add__(self, other):
        return FunctionResult(Arithmetic.ADD, self, other)
    def __sub__(self, other):
        return FunctionResult(Arithmetic.SUB, self, other)
    def __mul__(self, other):
        return FunctionResult(Arithmetic.MUL, self, other)
    def __rmul__(self, other):
        return FunctionResult(Arithmetic.MUL, self, other)
    def __div__(self, other):
        return FunctionResult(Arithmetic.DIV, self, other)
    def __pow__(self, other):
        return FunctionResult(Arithmetic.POW, self, other)
    def __pos__(self):
        return FunctionResult(Arithmetic.POS, self)
    def __neg__(self):
        return FunctionResult(Arithmetic.NEG, self)


class FunctionResult(Arithmetic):
    def __init__(self, fun, *args):
        self._fun = fun
        self._args = args

    def __str__(self):
        return "FunctionResult({})".format(self._fun)

    def __repr__(self):
        return str(self)

    def get_fun(self):
        return self._fun

    def get_name(self):
        return self._fun.get_name()

    def get_args(self):
        return self._args

    def get_act(self):
        return self._fun.get_act()

    def __hash__(self):
        return hash((self._fun, self._args))

    def __eq__(self, x):
        if isinstance(x, FunctionResult):
            return \
                self._fun == x._fun and \
                self._args == x._args
        return False




def function_configure(use_batch_norm=False, weight_factor=1.0):
    cfg = Config()
    cfg.use_batch_norm = use_batch_norm
    cfg.weight_factor = weight_factor
    return cfg

class Function(object):
    @classmethod
    def configure(cls, **kwargs):
        cls._CONFIG = function_configure(**kwargs)

    _CONFIG = function_configure()


    def __init__(self, name, *parents, **kwargs):
        self._act = kwargs.get("act")
        self._batch_norm = kwargs.get("batch_norm")
        self._name = name
        self._parent_funs = list(parents)
        self._config = Function._CONFIG.copy()

    def get_config(self):
        return self._config

    def __str__(self):
        return "Function({})".format(self.get_name())

    def __repr__(self):
        return str(self)

    def get_name(self):
        return self._name

    def get_act(self):
        return self._act

    def __or__(self, funs):
        if is_sequence(funs):
            for f in funs:
                self._parent_funs.append(f)
        else:
            self._parent_funs.append(funs)
        return self

    def __call__(self, *args):
        if len(self._parent_funs) > 0:
            return FunctionResult(self, *[ FunctionResult(pf, *args) for pf in self._parent_funs])
        else:
            return FunctionResult(self, *args)

    def __hash__(self):
        return hash((self._act, self._name, tuple(self._parent_funs)))

    def __eq__(self, x):
        if isinstance(x, Function):
            return \
                self._act == x._act and \
                self._name == x._name and \
                self._parent_funs == x._parent_funs
        return False


relu = BasicFunction("relu")
linear = BasicFunction("linear")
softplus = BasicFunction("softplus")
elu = BasicFunction("elu")
tanh = BasicFunction("tanh")
sigmoid = BasicFunction("sigmoid")


class LogFunction(BasicFunction):
    def __init__(self):
        super(LogFunction, self).__init__("log")

    def __call__(self, x):
        from vilab.api.model import Probability

        if isinstance(x, Probability):
            x.set_log_form(True)
            return x
        return super(LogFunction, self).__call__(x)

log = LogFunction()


