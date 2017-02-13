
import numbers
from variable import Variable

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
	pass

class Sequence(object):
    def __init__(self, name):
        self._name = name
        self._parts = {}

    def __str__(self):
        return "Sequence({})".format(self._name)

    def __repr__(self):
        return str(self)


    def __getitem__(self, key):
    	assert isinstance(key, Index), "Access to part of sequence must me done through Index class instance"
    	if not key in self._parts:
    		self._parts[key] = PartOfSequence(self._name + "[" + key.get_full_name() + "]")
    	return self._parts[key]

