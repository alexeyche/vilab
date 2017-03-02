
import logging
from ..util import is_sequence
from ..config import Config
from token import Token

class BasicFunction(Token):
    def __init__(self, name):
        super(BasicFunction, self).__init__(name)

    def __call__(self, *args):
        return FunctionResult(self, *args)


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


class FunctionResult(Token, Arithmetic):
    def __init__(self, fun, *args):
        super(FunctionResult, self).__init__(fun.get_name(), *args)
        self._fun = fun

    def get_fun(self):
        return self._fun

    def get_act(self):
        return self._fun.get_act()

    def __hash__(self):
        return hash((self._fun, self.get_args()))

    def __eq__(self, x):
        if isinstance(x, FunctionResult):
            return \
                self._fun == x._fun and \
                self.get_args() == x.get_args()
        return False




def function_configure(use_batch_norm=False, weight_factor=1.0):
    cfg = Config()
    cfg.use_batch_norm = use_batch_norm
    cfg.weight_factor = weight_factor
    return cfg

class Function(Token):
    @classmethod
    def configure(cls, **kwargs):
        cls._CONFIG = function_configure(**kwargs)

    _CONFIG = function_configure()


    def __init__(self, name, *parents, **kwargs):
        super(Function, self).__init__(name)
        self._act = kwargs.get("act")
        self._batch_norm = kwargs.get("batch_norm")
        self._parent_funs = list(parents)
        self._config = Function._CONFIG.copy()

    def get_config(self):
        return self._config

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
