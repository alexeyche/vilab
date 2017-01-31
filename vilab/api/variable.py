
import logging

class Variable(object):
    VARIABLE_DICT = {}

    def __init__(self, name):
        self._name = name
        self._requested_dependencies = []
        
        assert not self._name in Variable.VARIABLE_DICT, "Redefinition of variable {}".format(name)
        Variable.VARIABLE_DICT[self._name] = self
        self._model = None

    def get_model(self):
        assert not self._model is None, "Trying to get variable {} without model".format(self)
        return self._model

    def get_name(self):
        return self._name

    def __or__(self, x):
        self._requested_dependencies = [x]
        return self

    def __str__(self):
        return "Variable({})".format(self._name)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.get_name())

    def __eq__(self, x):
        if isinstance(x, Variable):
            return self.get_name() == x.get_name()
        else:
            raise Exception("Comparing with the strange type")

