

import logging
from collections import defaultdict
from vilab.api.variable import Variable
from vilab.api.density import Density, Unknown, Point
from vilab.api.function import FunctionCallee

class ModelSlice(object):
    def __init__(self, model, output, dependencies):
        self._output = output
        self._dependencies = dependencies
        self._model = model

    def __str__(self):
        return "ModelSlice {}({} | {})".format(
            self._model.get_name(), 
            self._output, 
            ",".join([str(d) for d in self._dependencies])
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, x):
        if isinstance(x, Density):
            logging.info("Describing variable {} with density {} in the context of the model {}".format(self._output, x, self._model))
            self._model.register_density(self._output, x)
            return True
        elif isinstance(x, FunctionCallee):
            logging.info("Describing variable {} with function {} in the context of the model {}".format(self._output, x, self._model))
            self._model.register_density(self._output, Point(x))
            return True
        elif isinstance(x, ModelSlice):
            return \
                self._output == x._output and \
                self._dependencies == self._dependencies and \
                self._model == self._model
        else:
            raise Exception("Comparing with the strange type: {}".format(x))

    def get_slice_info(self):
        return self._model, self._output, self._dependencies



class VariableRecord(object):
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            self.density = kwargs.get("density", Unknown())
            self.dependencies = kwargs.get("dependencies", [])
        if len(args) >= 1:
            self.density = args[0]
        if len(args) == 2:
            self.dependencies = args[1]
                

    def __str__(self):
        return "VariableRecord({}, {})".format(self.density, self.dependencies)

    def __repr__(self):
        return str(self)


class Model(object):
    def __init__(self, name):
        self._name = name
        self._var_records = defaultdict(VariableRecord)

    def __str__(self):
        return "Model({})".format(self._name)

    def __repr__(self):
        return str(self)

    def get_name(self):
        return self._name

    def __call__(self, *args):
        if len(args) > 1:
            v = args[0]
            assert isinstance(v, Variable), "Expecting variables as input to model"
            assert len(v._requested_dependencies) > 0, "Need to specify dependencies for 1 variable of model call"
            
            for a in args[1:]:
                assert isinstance(a, Variable), "Expecting variables as input to model"
                assert len(a._requested_dependencies) == 0, "Conditions for first variables are supported for now"

            dependencies = v._requested_dependencies + list(args[1:])
            v._requested_dependencies = list()
            self._var_records[v].dependencies = set(dependencies)
            
            logging.info("{} depends on {}".format(v, dependencies))
            return ModelSlice(self, v, dependencies)
        else:
            v = args[0]
            dependencies = []
            assert isinstance(v, Variable), "Expecting variables as input to model"
            if len(v._requested_dependencies) > 0:
                dependencies = v._requested_dependencies

                v._requested_dependencies = list()
                self._var_records[v].dependencies = set(dependencies)
                logging.info("{} depends on {}".format(v, dependencies))
            else:
                self._var_records[v].dependencies = set()
                logging.info("{} is unconditioned".format(v))
            return ModelSlice(self, v, dependencies)

    def __hash__(self):
        return hash(self.get_name())

    def __eq__(self, other):
        return other.get_name() == self.get_name()

    def register_density(self, var, pdf):
        logging.info("{}: {} == {}".format(self, var, pdf))
        self._var_records[var].density = pdf

    def get_variable_record(self, var):
        return self._var_records[var]


