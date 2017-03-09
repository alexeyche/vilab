
from vilab.api import *


class Engine(object):
    def __init__(self, name):
        self._name = name
        self._callbacks = {
            Variable: self.variable,
            Function: self.function,
            BasicFunction: self.basic_function,
        }

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Engine({})".format(self._name)

    def __call__(self, element, *args):
        logging.debug("{}: met element `{}`".format(self, element))

        callback = None
        
        strong_type_callbacks = [ v for k, v in self._callbacks.iteritems() if type(element) == k]
        inherit_type_callbacks = [ v for k, v in self._callbacks.iteritems() if isinstance(element, k)]
        
        if len(strong_type_callbacks) > 0:
            assert len(strong_type_callbacks) == 1
            callback = strong_type_callbacks[0]
        elif len(inherit_type_callbacks) > 0:
            assert len(inherit_type_callbacks) == 1
            callback = inherit_type_callbacks[0]
        else:
            raise Exception("Can't find callback for element: {}".format(element))
        
        res = callback(element, *args)
        logging.debug("{}: result {}".format(self, res))
        return res


    def variable(self, element, *args):
        raise NotImplementedError()

    def function(self, element, *args):
        raise NotImplementedError()

    def basic_function(self, element, *args):
        raise NotImplementedError()
