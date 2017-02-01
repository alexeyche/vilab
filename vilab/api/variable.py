
import logging

from density import *

class Variable(object):
    REGISTER = set()

    def __init__(self, name):
        assert not name in Variable.REGISTER, "Variable with name {} already defined".format(name)
        Variable.REGISTER.add(name)
        self._name = name
        self._requested_dependencies = []
        self._dependencies = set()
        self._density = Unknown()
        self._generated_by_models = list()
        self._model = None

    def get_density(self):
        return self._density

    def get_dependencies(self):
        return self._dependencies

    def set_density(self, d):
        self._density = d


    def set_model(self, model):
        assert not model in self._generated_by_models, "Variable {} is already generated by model {}".format(self, model)
        self._generated_by_models.append(model)
        self._model = model

    def get_model(self):
        return self._model
        
        # assert len(self._generated_by_models) > 0, "Variable {} is not generated by any model"
        # assert len(self._generated_by_models) == 1, "Variable {} is generated by too many models"
        
        # return self._generated_by_models[0]

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
        return hash((self._name, self._density))

    def __eq__(self, x):
        if isinstance(x, Variable):
            return \
                self.get_name() == x.get_name() and \
                self.get_density() == x.get_density() and \
                self.get_dependencies() == x.get_dependencies()
        else:
            raise Exception("Comparing with the strange type: {}".format(x))

