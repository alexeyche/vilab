
import logging

from token import Token

class Variable(Token):
    REGISTER = {}

    def __init__(self, name):
        super(Variable, self).__init__(name)

        assert not name in Variable.REGISTER, "Variable with name {} already defined".format(name)
        Variable.REGISTER[name] = self
        self._requested_dependencies = []
        
        self._models = tuple()     # Just for convinience of deduction of deduce(x), 
                                   # but it will fail if variable is described by several densities (condition on something)
        self._descriptive = True


    def is_descriptive(self):
        return self._descriptive

    def set_non_descriptive(self):
        self._descriptive = False

    def set_model(self, model):
        self._models += (model,)

    def get_models(self):
        return self._models

    def __or__(self, x):
        self._requested_dependencies = [x]
        return self

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, x):
        from density import Density
        from function import FunctionResult
        from model import Model

        if isinstance(x, Variable):
            return self.get_name() == x.get_name()
        elif self._descriptive and (isinstance(x, Density) or isinstance(x, FunctionResult)):
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
