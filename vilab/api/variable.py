
import logging


class Variable(object):
    REGISTER = set()

    def __init__(self, name):
        assert not name in Variable.REGISTER, "Variable with name {} already defined".format(name)
        Variable.REGISTER.add(name)
        self._name = name
        self._requested_dependencies = []
        
        self._models = tuple()     # Just for convinience of deduction of deduce(x), 
                                   # but it will fail if variable is described by several densities (condition on something)

    def set_model(self, model):
        self._models += (model,)

    def get_models(self):
        return self._models
        
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
        return hash(self._name)

    def __eq__(self, x):
        from density import Density
        from function import FunctionResult
        from model import Model

        if isinstance(x, Variable):
            return self.get_name() == x.get_name()
        elif isinstance(x, Density) or isinstance(x, FunctionResult):
            p_tmp = Model(self.get_name())
            return p_tmp(self | None) == x  # unknown dependencies
        else:
            return False

    def __cmp__(self, x):
        assert isinstance(x, Variable)
        if self == x:
            return 0
        return -1 if self.get_name() < x.get_name() else 1

    def get_scope_name(self):
        n = self.get_name()
        return n.replace("[", "_").replace("]", "")
