
import numbers

from vilab.log import setup_log
import logging

from vilab.api import *

from collections import namedtuple
from collections import OrderedDict
from collections import deque

def get_zero_offset(element):
    t_elems = [ v for k, v in element.get_seq().get_parts().iteritems() if isinstance(k, Index) and k.get_offset() == 0 ]
    assert len(t_elems) > 0, "Need to define indexed sequence with zero offset for {}".format(element.get_seq())
    assert len(t_elems) == 1, "Too many zero offset indexed sequences for {}".format(element.get_seq())
    t_elem = t_elems[0]
    return t_elem

def print_tree(d, level=0):
    ident = "".join(["  "]*level)
    res = ""
    if level == 0:
        res += "\n"
    for k, v in d.iteritems():
        if isinstance(v, OrderedDict):
            res += "{}{} -> \n".format(ident, k)
            res += print_tree(v, level+1)
        else:
            res += "{}{} -> {}\n".format(ident, k, v)
    return res            


class Parser(object):
    Ctx = namedtuple("Ctx", [
        "statement_id", 
        "dependencies",
        "variables_tree"
    ])

    SeqInfo = namedtuple("SeqCtx", [
        "init",
        "output",
        "current",
        "previous"
    ])
    
    @staticmethod
    def get_ctx_with(
        ctx, 
        statement_id=None,
        dependencies=None,
        variables_tree=None
    ):
        return Parser.Ctx(
            ctx.statement_id if statement_id is None else statement_id,
            ctx.dependencies if dependencies is None else dependencies,
            ctx.variables_tree if variables_tree is None else variables_tree
        )


    def __init__(self):
        self._level = 0
        self._callbacks = {
            Variable: self._variable,
            Probability: self._probability,
            PartOfSequence: self._part_of_sequence,
        }

    
    def deduce(self, element):
        ctx = Parser.Ctx(None, None, OrderedDict())
        
        leaves = set()
        
        sequences = []
        seq_info = Parser.SeqInfo(set(), set(), set(), set())
        
        def collect_info(element, *args):
            if len(args) == 0:
                leaves.add(element)
            
            if isinstance(element, PartOfSequence):
                idx = element.get_idx()
                if isinstance(idx, numbers.Integral):
                    if idx == 0:
                        seq_info.init.add(element)
                    elif idx == -1:
                        seq_info.output.add(element)
                    else:
                        raise Exception("Unsupported index for sequence: {}".format(idx))
                elif isinstance(idx, Index):
                    if idx.get_offset() == 0:
                        seq_info.current.add(element)
                    elif idx.get_offset() == -1:
                        seq_info.previous.add(element)
                    else:
                        raise Exception("Unsupported index offset: {}".format(idx))
            return element

        result = self._deduce(element, ctx, collect_info)
        if logging.getLogger().level == logging.DEBUG:
            logging.debug("== POSTPROCESS =================================")
        
        for prev in seq_info.previous:
            try:
                prev_gen = get_zero_offset(prev)
            except:
                raise Exception("Failed to find generation process for state variable: {}".format(prev))
            logging.debug("Finding out generating method for {}".format(prev))
            self._deduce(prev_gen, ctx, collect_info)
        
        if logging.getLogger().level == logging.DEBUG:
            logging.debug("Resulting variables tree: \n{}".format(print_tree(ctx.variables_tree)))

        self._deduplicate_variables_tree(ctx.variables_tree)
        if logging.getLogger().level == logging.DEBUG:
            logging.debug("Resulting variables tree after dedup: \n{}".format(print_tree(ctx.variables_tree)))

        logging.debug(
            "Sequence releated info:\n\n"
            "\tinit: {}\n"
            "\tcurrent: {}\n"
            "\tprevious: {}\n"
            "\toutput: {}\n"
            .format(seq_info.init, seq_info.current, seq_info.previous, seq_info.output)
        )
            

        self._deduce_sequences(seq_info, ctx.variables_tree)
    
    def _deduplicate_variables_tree(self, variables_tree):
        def find_depth(d, level=0):
            acc = 0
            for k, v in d.iteritems():
                if isinstance(v, OrderedDict):
                    acc += find_depth(v, level+1)
                else:
                    acc += level
            return acc

        def dedup(dest):
            max_key, max_depth = None, None
            for k, v in dest.iteritems():
                if isinstance(v, OrderedDict):
                    d = find_depth(v)
                    if max_depth is None or d > max_depth:
                        max_depth = d
                        max_key = k

            if max_key is None:
                return 

            print "max_key: {}".format(max_key)
            
            def find_dups(d):
                for k, v in d.iteritems():
                    if k != max_key and k in dest:
                        if type(v) == type(dest[k]):
                            assert dest[k] == v, "Found that variable {} can be deduced by several different pathways".format(k)
                            logging.debug("Getting rid of duplicate chain for {}".format(k))
                            del dest[k]
                        else:
                            print "{} != {}".format(v, dest[k])
                    elif isinstance(v, OrderedDict):
                        find_dups(v)
            
            find_dups(dest[max_key])
            
            if isinstance(dest[max_key], OrderedDict):
                dedup(dest[max_key])

        dedup(variables_tree)

    def _deduce_sequences(self, seq_info, variables_tree):
        def check_inits(d):
            for k, v in d.iteritems():
                if isinstance(k, PartOfSequence):
                    if  k.get_idx() == 0 and not v is None: # there is dependency on other stuff
                        pass
                    elif isinstance(k.get_idx(), Index) and k.get_idx().get_offset() == 0 and v is None: # it's input from data to NN
                        pass


    def _deduce(self, element, ctx, engine):
        logging.debug("Deducing element `{}`".format(element))

        self._level += 1

        if logging.getLogger().level == logging.DEBUG:
            setup_log(logging.DEBUG, ident_level=self._level)
        
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
            callback = self._default_callback
            
        result = callback(element, ctx, engine)
        
        self._level -= 1

        if logging.getLogger().level == logging.DEBUG:
            setup_log(logging.DEBUG, ident_level=self._level)

        logging.debug("Done: {}".format(element))
        return result

    def _default_callback(self, element, ctx, engine):
        args_result = []
        for elem in element.get_args():
            args_result.append(self._deduce(elem, ctx, engine))
        return engine(element, *args_result)

    def _probability(self, element, ctx, engine):
        next_element = element.get_args()
        assert len(next_element) == 1, "Can't find specification of {}".format(element)
        args_result = self._deduce(
            next_element[0], 
            Parser.get_ctx_with(
                ctx, 
                statement_id=element.get_statement_id(), 
                dependencies=element.get_dependencies()
            ), 
            engine
        )
        return engine(element, args_result)

    def _variable(self, element, ctx, engine):
        candidates = []
        
        for m in element.get_models():
            for var_statement_id, density, probability in m.get_var_probabilities(element):
                min_global_statement = ctx.statement_id is None or var_statement_id <= ctx.statement_id  # intersected with current statement id
                
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
            var_dict = OrderedDict()
            arg_result = self._deduce(
                next_density, 
                Parser.get_ctx_with(
                    ctx,
                    statement_id=next_statement_id, 
                    dependencies=next_probability.get_dependencies(),
                    variables_tree=var_dict
                ), 
                engine
            )
            ctx.variables_tree[element] = var_dict
            return engine(element, arg_result)
        ctx.variables_tree[element] = None
        return engine(element)
        
    def _part_of_sequence(self, element, ctx, engine):
        idx = element.get_idx()
        if idx == 0:
            return self._variable(element, ctx, engine)
        elif idx == -1:
            t_elem = get_zero_offset(element)
            ret = self._deduce(t_elem, ctx, engine)
            return engine(element, ret)
        elif isinstance(idx, Index):
            seq = element.get_seq()
            if idx.get_offset() == -1: # t-1
                seq_init = self._deduce(seq[0], ctx, engine)
                return engine(element, seq_init)
            elif idx.get_offset() == 0: # t
                seq_t = self._variable(element, ctx, engine)
                return seq_t
            else:
                raise Exception("Unsupported index offset: {}".format(idx))
        else:
            raise Exception("Unsupported indexing {}".format(idx))