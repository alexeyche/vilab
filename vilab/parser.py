
from vilab.log import setup_log
import logging

from vilab.api import *

from collections import namedtuple

class Parser(object):
    Ctx = namedtuple("Ctx", [
        "statement_id", 
        "dependencies",
    ]) 

    def __init__(self):
        self._level = 0
        self._callbacks = {
            Variable: self._variable,
            Probability: self._probability
        }

    
    def deduce(self, element):
        ctx = Parser.Ctx(None, None)
        
        leaves = set()
        
        def collect_leaves(element, *args):
            if len(args) == 0:
                leaves.add(element)
            return element

        result = self._deduce(element, ctx, collect_leaves)
        print leaves

    def _deduce(self, element, ctx, engine):
        logging.debug("Deducing element `{}`".format(element))

        self._level += 1

        if logging.getLogger().level == logging.DEBUG:
            setup_log(logging.DEBUG, ident_level=self._level)
        
        callback = None

        callbacks = [ v for k, v in self._callbacks.iteritems() if isinstance(element, k)]
        if len(callbacks) > 0:
            assert len(callbacks) == 1
            callback = callbacks[0]
        else:
            callback = self._default_callback
            
        result = callback(element, ctx, engine)
        
        self._level -= 1

        if logging.getLogger().level == logging.DEBUG:
            setup_log(logging.DEBUG, ident_level=self._level)

        logging.debug("Done: {}".format(element))

    def _default_callback(self, element, ctx, engine):
        args_result = []
        for elem in element.get_args():
            args_result.append(self._deduce(elem, ctx, engine))
        return engine(element, *args_result)

    def _probability(self, element, ctx, engine):
        next_element = element.get_args()
        assert len(next_element) == 1
        args_result = self._deduce(next_element[0], Parser.Ctx(element.get_statement_id(), element.get_dependencies()), engine)
        return engine(element, args_result)

    def _variable(self, element, ctx, engine):
        candidates = []
        
        for m in element.get_models():
            for var_statement_id, density, probability in m.get_var_probabilities(element):
                min_global_statement = ctx.statement_id is None or var_statement_id < ctx.statement_id  # intersected with current statement id
                
                if min_global_statement:
                    candidates.append((var_statement_id, density, probability))

        next_density, next_statement_id, next_probability = None, None, None
        if len(candidates) == 1:
            logging.debug("Got 1 candidate {} to describe {}. Easy choice".format(candidates[0], element))
            next_statement_id, next_density, next_probability = candidates[0]
        
        elif len(candidates) > 1:
            logging.debug("Got {} candidates to describe variable {}, need to choose ...".format(len(candidates), element))
            assert not ctx.dependencies is None, "Got too many descriptions of variable {}: {}; failing to choose one of them (need more context)" \
                .format(element, ", ".join([str(c[2]) for c in candidates]))

            deps = set([d for d in ctx.dependencies if d != element])
            if len(deps) == 0:
                logging.debug("Looking for description that is unconditioned")

                candidates = [ (c_st, c_dens, c_prob) for c_st, c_dens, c_prob in candidates if len(c_prob.get_dependencies()) == 0]
                
                assert len(candidates) > 0, "Failed to find any description of {} which includes unconditioned dependency".format(element)
                assert len(candidates) == 1, "Got too many descriptions of variable {} which includes unconditioned dependency".format(element)                    
            else:
                logging.debug("Looking for description that has {} as subset".format(deps))
                
                candidates = [ (c_st, c_dens, c_prob) for c_st, c_dens, c_prob in candidates if deps <= set(c_prob.get_dependencies())]
            
                assert len(candidates) > 0, "Failed to find any description of {} which has dependencies that includes {}".format(element, deps)
                assert len(candidates) == 1, "Got too many descriptions of variable {} which has dependencies that includes {}".format(element, deps)
            
            logging.debug("Found this one: {}".format(candidates[0][2]))
                
            next_statement_id, next_density, next_probability = candidates[0]

        if not next_density is None:
            arg_result = self._deduce(next_density, Parser.Ctx(next_statement_id, next_probability.get_dependencies()), engine)
            return engine(element, arg_result)
        return engine(element)
        
        