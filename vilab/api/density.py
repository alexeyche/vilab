
import logging
from function import Arithmetic

def get_name_if_possible(obj):
    if hasattr(obj, "get_name"):
        return obj.get_name()
    return str(obj)

class Density(object):
    def __init__(self, name, *args, **kwargs):
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
        

class N0(N):
    def __init__(self):
        super(N, self).__init__("Normal0", 0.0, 0.0)

class Cat(Density):
    def __init__(self, pi):
        super(Cat, self).__init__("Categorical", pi)


class Unknown(Density):
    def __init__(self):
        super(Unknown, self).__init__("Unknown")

class Point(Density):
    def __init__(self, point):
        super(Point, self).__init__("Point", point)



class Metrics(Arithmetic):
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name
    
    def __str__(self):
        return "Metrics({})".format(self.get_name())

    def __repr__(self):
        return str(self)



class KL(Metrics):
    def __init__(self, p, q):
        super(KL, self).__init__("KL")
        self._p = p
        self._q = q