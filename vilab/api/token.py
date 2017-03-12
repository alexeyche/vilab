
from collections import defaultdict
import numbers

class TokenMeta(type):
    def __new__(meta, name, bases, dct):
        # "Allocating memory for class", name
        return super(TokenMeta, meta).__new__(meta, name, bases, dct)

    def __init__(cls, name, bases, dct):
        # "Initializing class", name
        super(TokenMeta, cls).__init__(name, bases, dct)

        cls.class_name = name
        cls.register = defaultdict(list)

    def __call__(cls, *args, **kwds):
        obj = type.__call__(cls, *args, **kwds)
        cls.register[obj.get_name()].append(obj)
        return obj




class Token(object):
    __metaclass__ = TokenMeta

    def __init__(self, name, *args):
        self._name = name
        args_list = []
        for a in args:
            if isinstance(a, numbers.Integral) or isinstance(a, numbers.Real):
                args_list.append(IntegralType(a))
            else:
                args_list.append(a)
        self._args = tuple(args_list)
    
    def __str__(self):
        return "{}({})".format(self.get_class_name(), self.get_name())

    def __repr__(self):
        return str(self)

    def get_name(self):
        return self._name

    def get_class_name(self):
        return self.__class__.__name__

    def get_args(self):
        return self._args

    def is_empty_args(self):
        return len(self._args) == 0

    def set_args(self, args):
        self._args = args

class IntegralType(Token):
    def __init__(self, value):
        super(IntegralType, self).__init__(type(value).__name__)
        self._value = value

    def __hash__(self):
        return hash(self._value)

    def __eq__(self, other):
        return self._value == other._value
        
    def get_value(self):
        return self._value
        
