
import numbers
from variable import Variable
from function import Arithmetic
from token import Token

class Index(Token):
    def __init__(self, name, offset = 0):
        super(Index, self).__init__(Index.get_full_name(name, offset))
        self._offset = offset
        
    @staticmethod
    def get_full_name(name, offset):
        descr = ""
        if offset > 0:
            descr = "+{}".format(offset)
        elif offset < 0:
            descr = "-{}".format(abs(offset))
        return "{}{}".format(name, descr)

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
        if isinstance(other, Index):
            return self._name == other._name and self._offset == other._offset
        return False

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

    def __hash__(self):
        return hash((self._seq, self._idx))


class Sequence(Token):
    REGISTER = {}

    def __init__(self, name):
        super(Sequence, self).__init__(name)
        assert not name in Sequence.REGISTER, "Sequence with name {} already defined".format(name)

        self._parts = {}

        Sequence.REGISTER[name] = self

    def __hash__(self):
        return hash(self.get_name())

    def __eq__(self, other):
        return self.get_name() == other.get_name()

    def __getitem__(self, key):
        assert isinstance(key, Index) or isinstance(key, numbers.Integral), "Access to part of sequence must be done through Index class instance or through integral numbers"
        if not key in self._parts:
            if isinstance(key, numbers.Integral):
                key_name = str(key)
            else:
                key_name = key.get_name()

            self._parts[key] = PartOfSequence(self, key, self._name + "[" + key_name + "]")
        return self._parts[key]

    def get_parts(self):
        return self._parts

class SequenceOperation(Token, Arithmetic):
    def __init__(self, name, expr):
        super(SequenceOperation, self).__init__(name, expr)


class Summation(SequenceOperation):
    def __init__(self, expr):
        super(Summation, self).__init__("Summation", expr)

class Iterate(SequenceOperation):
    def __init__(self, expr):
        super(Iterate, self).__init__("Iterate", expr)

