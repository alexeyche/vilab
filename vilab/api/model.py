

import logging
from collections import defaultdict
from vilab.api.variable import Variable
from vilab.api.density import Density, Unknown, DiracDelta

from token import Token



class Probability(Token):
    def __init__(self, model, output, dependencies):
        deps_list = [] if dependencies is None else dependencies
        super(Probability, self).__init__("{}({}{}{})".format(
            model.get_name(),
            output.get_name(), 
            "" if len(deps_list) == 0 else " | ",
            ", ".join([d.get_name() for d in deps_list])
        ))
        
        self._output = output
        self._dependencies = None if dependencies is None else tuple(sorted(dependencies))
        self._model = model
        self._log_form_flag = False
        
    # def __str__(self):
    #     deps = [] if self._dependencies is None else self._dependencies
    #     return "{}{}({}{}{}){}".format(
    #         ("" if not self._log_form_flag else "log("),
    #         self._model.get_name(), 
    #         self._output.get_name(), 
    #         "" if len(deps) == 0 else " | ",
    #         ", ".join([d.get_name() for d in deps]),
    #         ("" if not self._log_form_flag else ")")
    #     )

    def get_context_name(self):
        s = str(self)
        return s \
            .replace("(", "/") \
            .replace(")", "/") \
            .replace("|", "_") \
            .replace(" ", "") \
            .replace(",", "_") \
            .replace("[", "") \
            .replace("]", "")

    def __eq__(self, x):
        from vilab.api.function import FunctionResult

        if isinstance(x, Density) and self._output.is_descriptive():
            logging.info("Describing variable {} with density {} in the context of {}".format(self._output, x, self._model))
            self._output.set_model(self._model)
            self._model.save_description(self._output, self._dependencies, x)
            return True
        elif isinstance(x, FunctionResult) and self._output.is_descriptive():
            logging.info("Describing variable {} with function {} in the context of {}".format(self._output, x, self._model))
            self._output.set_model(self._model)
            self._model.save_description(self._output, self._dependencies, DiracDelta(x))
            return True
        elif isinstance(x, Probability):
            return \
                self._output == x._output and \
                self._dependencies == x._dependencies and \
                self._model == x._model
        else:
            return False

    def get_components(self):
        return self._model, self._output, self._dependencies

    def get_output(self):
        return self._output

    def set_log_form(self, log_form_flag):
        self._log_form_flag = log_form_flag

    def is_log_form(self):
        return self._log_form_flag

    def get_dependencies(self):
        return self._dependencies

    def get_density(self):
        statement_id, density = self._model.get_description(self._output, self._dependencies)
        if not density is None:
            return density
        raise Exception("Failed to deduce density for {}".format(self))

    def get_output(self):
        return self._output

    def get_model(self):
        return self._model

    def __hash__(self):
        return hash((self._model, self._output, self._dependencies))

    def get_statement_id(self):
        descr = self._model.get_description(self._output, self._dependencies)
        assert not descr is None, "Trying to get unknown statemet: {}".format(self)
        return descr[0]
        
    def get_args(self):
        descr = self._model.get_description(self._output, self._dependencies)
        if descr is None:
            return []
        return [descr[1]]


class Model(Token):
    STATEMENTS_NUM = 0

    def __init__(self, name):
        super(Model, self).__init__(name)
        self._descriptions = {}

    def get_description(self, var, deps):
        key = (var, deps)
        return self._descriptions.get(key)

    def get_var_probabilities(self, var):
        return tuple([ (v[0], v[1], Probability(self, var, k[1]))  for k, v in self._descriptions.iteritems() if k[0] == var ])

    def save_description(self, var, deps, density):
        key, val = (var, deps), (Model.STATEMENTS_NUM, density)
        assert not key in self._descriptions, \
            "Trying to redefine variable {} conditioned on dependencies {} with density {}".format(var, deps, density)
        self._descriptions[key] = val
        
        Model.STATEMENTS_NUM += 1

    def has_description(self, var, deps):
        return (var, deps) in self._descriptions

    def __call__(self, *args):
        v = args[0]
        assert isinstance(v, Variable), "Expecting variables as input to model"

        if len(args) > 1:
            assert len(v._requested_dependencies) > 0, "Need to specify dependencies for 1 variable of model call"
            
            for a in args[1:]:
                assert isinstance(a, Variable), "Expecting variables as input to model"
                assert len(a._requested_dependencies) == 0, "Conditions for first variables are supported for now"

            unknown = len(v._requested_dependencies) == 1 and v._requested_dependencies[0] is None
            
            dependencies = tuple(v._requested_dependencies + list(args[1:])) if not unknown else None
            v._requested_dependencies = list()
            v._dependencies = dependencies
            
            logging.info("{} depends on {}".format(v, dependencies))
            return Probability(self, v, dependencies)
        else:
            dependencies = tuple()
            if len(v._requested_dependencies) > 0:
                unknown = len(v._requested_dependencies) == 1 and v._requested_dependencies[0] is None
            
                dependencies = tuple(v._requested_dependencies) if not unknown else None

                v._requested_dependencies = list()
                v._dependencies = dependencies
                logging.info("{} depends on {}".format(v, dependencies))
            else:
                v._dependencies = tuple()
                logging.info("{} is unconditioned".format(v))
            return Probability(self, v, dependencies)

    def __hash__(self):
        return hash(self.get_name())

    def __eq__(self, other):
        return other.get_name() == self.get_name()


