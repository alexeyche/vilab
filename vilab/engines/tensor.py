
from vilab.util import is_sequence

class Tensor(object):
	def __init__(self, data, shape):
		self._data = data
		self._shape = shape

	def __repr__(self):
		return str(self)

	def __str__(self):
		return "Tensor({})".format("x".join([str(s) for s in self._shape]))

	def get_shape(self):
		return self._shape