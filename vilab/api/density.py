
import logging
from function import Arithmetic
from ..config import Config

def get_name_if_possible(obj):
    if hasattr(obj, "get_name"):
        return obj.get_name()
    return str(obj)


def density_configure(importance_samples=1):
    cfg = Config()
    cfg.importance_samples = importance_samples
    return cfg

class Density(object):
    @classmethod
    def configure(cls, **kwargs):
        cls._CONFIG = density_configure(**kwargs)

    _CONFIG = density_configure()


    def __init__(self, name, *args):
        self._name = name
        self._args = list(args)
        self._config = Density._CONFIG.copy()

    def get_config(self):
        return self._config

    def __str__(self):
        return "{}({})".format(self._name, ",".join([ get_name_if_possible(a) for a in self._args]))

    def __repr__(self):
        return str(self)

    def get_args(self):
        return self._args

    def set_args(self, args):
        self._args = args

    def get_name(self):
        return self._name


class N(Density):
    def __init__(self, mu, logvar):
        super(N, self).__init__("Normal", mu, logvar)
        self._config = N._CONFIG.copy()
        

class B(Density):
    def __init__(self, logits):
        super(B, self).__init__("Binomial", logits)
        self._config = B._CONFIG.copy()


N0 = N(0.0, 0.0) 

class Cat(Density):
    def __init__(self, pi):
        super(Cat, self).__init__("Categorical", pi)
        self._config = Cat._CONFIG.copy()


class Unknown(Density):
    def __init__(self):
        super(Unknown, self).__init__("Unknown")
        self._config = Unknown._CONFIG.copy()

class DiracDelta(Density):
    def __init__(self, point):
        super(DiracDelta, self).__init__("DiracDelta", point)
        self._config = DiracDelta._CONFIG.copy()



class Metrics(Arithmetic):
    def __init__(self, name, *args):
        self._name = name
        self._args = list(args)

    def get_name(self):
        return self._name
    
    def __str__(self):
        return "Metrics({})".format(self.get_name())

    def __repr__(self):
        return str(self)

    def get_args(self):
        return self._args


class KL(Metrics):
    def __init__(self, p, q):
        super(KL, self).__init__("KL", p, q)
        self._p = p
        self._q = q

class SquaredLoss(Metrics):
    def __init__(self, p, q):
        super(SquaredLoss, self).__init__("SquaredLoss", p, q)
        self._p = p
        self._q = q
