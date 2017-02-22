
import numbers
from variable import Variable
from function import Arithmetic

class Index(object):
    def __init__(self, name, offset = 0):
        self._name = name
        self._offset = offset

    def __str__(self):
        return "Index({})".format(self.get_full_name())

    def get_full_name(self):
        descr = ""
        if self._offset > 0:
            descr = "+{}".format(self._offset)
        elif self._offset < 0:
            descr = "-{}".format(abs(self._offset))
        return "{}{}".format(self._name, descr)


    def __repr__(self):
        return str(self)

    def get_name(self):
        return self._name

    def get_offset(self):
        return self._offset

    def __add__(self, other):
        assert isinstance(other, numbers.Integral), \
            "Index offseting supported only for integral types, got {}".format(other)
        return Index(self.get_name(), self.get_offset() + other)

    def __sub__(self, other):
        assert isinstance(other, numbers.Integral), \
            "Index offseting supported only for integral types, got {}".format(other)
        return Index(self.get_name(), self.get_offset() - other)

    def __eq__(self, other):
        return self._name == other._name and self._offset == other._offset

    def __hash__(self):
        return hash((self._name, self._offset))

    def __cmp__(self, other):
        if self == other:
            return 0
        return -1


class PartOfSequence(Variable):
    def __init__(self, seq, idx, name):
        super(PartOfSequence, self).__init__(name)
        self._seq = seq
        self._idx = idx

    def get_idx(self):
        return self._idx

    def get_seq(self):
        return self._seq


class Sequence(object):
    REGISTER = {}

    def __init__(self, name):
        assert not name in Sequence.REGISTER, "Sequence with name {} already defined".format(name)

        self._name = name
        self._parts = {}

        Sequence.REGISTER[name] = self

    def __str__(self):
        return "Sequence({})".format(self._name)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        return self._name == other._name

    def __getitem__(self, key):
        assert isinstance(key, Index) or isinstance(key, numbers.Integral), "Access to part of sequence must be done through Index class instance or through integral numbers"
        if not key in self._parts:
            if isinstance(key, numbers.Integral):
                key_name = str(key)
            else:
                key_name = key.get_full_name()

            self._parts[key] = PartOfSequence(self, key, self._name + "[" + key_name + "]")
        return self._parts[key]

    def get_parts(self):
        return self._parts

    def get_name(self):
        return self._name

class SequenceOperation(Arithmetic):
    def __init__(self, name, expr):
        self._name = name
        self._expr = expr


    def __str__(self):
        return "{}({})".format(self._name, self._expr)

    def __repr__(self):
        return str(self)


    def get_expr(self):
        return self._expr

class Summation(SequenceOperation):
    def __init__(self, expr):
        super(Summation, self).__init__("Summation", expr)

