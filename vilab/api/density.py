
import logging
from function import Arithmetic

def get_name_if_possible(obj):
    if hasattr(obj, "get_name"):
        return obj.get_name()
    return str(obj)

class Density(object):
    def __init__(self, name, *args):
        self._name = name
        self._args = list(args)

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
        

class B(Density):
    def __init__(self, logits):
        super(B, self).__init__("Binomial", logits)


N0 = N(0.0, 0.0) 

class Cat(Density):
    def __init__(self, pi):
        super(Cat, self).__init__("Categorical", pi)


class Unknown(Density):
    def __init__(self):
        super(Unknown, self).__init__("Unknown")

class DiracDelta(Density):
    def __init__(self, point):
        super(DiracDelta, self).__init__("DiracDelta", point)



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
