
import numbers

from vilab.log import setup_log
import logging

from vilab.api import *

from collections import namedtuple
from collections import OrderedDict
from collections import deque
from engines.engine import Engine, CallbackEngine

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
        "variables_tree",
        "requested_shape",
        "scope",
        "density_view",
        "probability_output",
        "sequence_variables",
    ])

    SeqInfo = namedtuple("SeqCtx", [
        "init",
        "previous",
        "current",
        "output"
    ])
    
    def __init__(self, batch_size, structure={}, ):
        self._level = 0
        self._callbacks = {
            Variable: self._variable,
            Probability: self._probability,
            PartOfSequence: self._part_of_sequence,
            FunctionResult: self._function_result,
            Metrics: self._metrics,
            Density: self._density,
        }
        self._structure = structure
        self._batch_size = batch_size
        self._verbose = False

    @staticmethod
    def get_ctx_with(
        ctx, 
        statement_id=None,
        dependencies=None,
        variables_tree=None,
        requested_shape=None,
        scope=None,
        density_view=None,
        probability_output=None,
        sequence_variables=None,
    ):
        return Parser.Ctx(
            ctx.statement_id if statement_id is None else statement_id,
            ctx.dependencies if dependencies is None else dependencies,
            ctx.variables_tree if variables_tree is None else variables_tree,
            ctx.requested_shape if requested_shape is None else requested_shape,
            ctx.scope if scope is None else scope,
            ctx.density_view if density_view is None else density_view,
            ctx.probability_output if probability_output is None else probability_output,
            ctx.sequence_variables if sequence_variables is None else sequence_variables,
        )
    
    @staticmethod
    def get_default_ctx():
         return Parser.Ctx(None, None, OrderedDict(), None, "", Density.View.SAMPLE, None, OrderedDict())

    def parse(self, elements, engine):
        res = []
        
        ctx = Parser.get_default_ctx()
        seq_info = Parser.SeqInfo(set(), set(), set(), set())
        leaves = set()
        
        for elem in elements:
            res.append(
                self.deduce(elem, ctx, seq_info, leaves, engine)
            )
        return res

    def deduce(self, element, ctx, seq_info, leaves, engine):
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

        # result = self._deduce(element, ctx, CallbackEngine(collect_info))
        # if logging.getLogger().level == logging.DEBUG:
        #     logging.debug("== POSTPROCESS =================================")
        
        # for prev in seq_info.previous:
        #     try:
        #         prev_gen = get_zero_offset(prev)
        #     except:
        #         raise Exception("Failed to find generation process for state variable: {}".format(prev))
        #     logging.debug("Finding out generating method for {}".format(prev))
        #     self._deduce(prev_gen, ctx, collect_info)
        
        # if logging.getLogger().level == logging.DEBUG:
        #     logging.debug("Resulting variables tree: \n{}".format(print_tree(ctx.variables_tree)))

        # self._deduplicate_variables_tree(ctx.variables_tree)
        # if logging.getLogger().level == logging.DEBUG:
        #     logging.debug("Resulting variables tree after dedup: \n{}".format(print_tree(ctx.variables_tree)))

        # def log_seq(seq_info):
        #     logging.debug(
        #         "\n"
        #         "\tinit: {}\n"
        #         "\tcurrent: {}\n"
        #         "\tprevious: {}\n"
        #         "\toutput: {}\n"
        #         .format(seq_info.init, seq_info.current, seq_info.previous, seq_info.output)
        #     )    

        # logging.debug("Sequence related info before preprocess: ")
        # log_seq(seq_info)
        # seq_infos = self._deduce_sequences(seq_info, ctx.variables_tree)
        
        # logging.debug("Sequence related info after preprocess: ")
        # for si in seq_infos:
        #     log_seq(si)

        ctx = Parser.get_default_ctx()
        
        return self._deduce(element, ctx, engine)
            
    def _deduplicate_variables_tree(self, variables_tree):
        def find_depth(d, level=0):
            max_level = level
            for k, v in d.iteritems():
                if isinstance(v, OrderedDict):
                    max_level = max(max_level, find_depth(v, level+1))
            return max_level

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

            def find_dups(d, level=0):
                for k, v in d.iteritems():
                    if k != max_key and k in dest and type(v) == type(dest[k]):
                        assert dest[k] == v, "Found that variable {} can be deduced by several different pathways".format(k)
                        logging.debug("Getting rid of duplicate chain for {}".format(k))
                        del dest[k]
                    
                    if isinstance(v, OrderedDict):
                        find_dups(v, level+1)

            find_dups(dest[max_key])
            
            if isinstance(dest[max_key], OrderedDict):
                dedup(dest[max_key])

        dedup(variables_tree)

    def _deduce_sequences(self, seq_info, variables_tree):
        def collect_sequences(d, seqs):
            for k, v in d.iteritems():
                go_deep = isinstance(v, OrderedDict)

                if isinstance(k, PartOfSequence):
                    seqs.add(k.get_seq())
                    if k.get_idx() == 0 and not v is None: # there is dependency on other stuff
                        yield seqs
                        if go_deep:
                            new_seqs = set()
                            for vs in collect_sequences(v, new_seqs):
                                yield vs

                    elif go_deep:
                        for vs in collect_sequences(v, seqs):
                            yield vs
                    else:
                        yield seqs

                elif go_deep:
                    for vs in collect_sequences(v, seqs):
                        yield vs
                else:
                    yield seqs
        
        splitted_seqs = []
        for k in collect_sequences(variables_tree, set()):
            k = sorted(k)
            if not k in splitted_seqs:
                splitted_seqs.append(k)
            
        
        out = []
        for seqs in splitted_seqs:
            seq_set = set(seqs)
            split_seq_info = Parser.SeqInfo(
                [ s for s in seq_info.init if s.get_seq() in seq_set ],
                [ s for s in seq_info.previous if s.get_seq() in seq_set ],
                [ s for s in seq_info.current if s.get_seq() in seq_set ],
                [ s for s in seq_info.output if s.get_seq() in seq_set ]
            )
            out.append(split_seq_info) 
        return out

    def _deduce(self, element, ctx, engine):
        if self._verbose:
            logging.debug("Deducing element: \n elem: {},\n ctx: \n\t{}".format(
                element, 
                "\n\t".join(
                    ["{} -> {}".format(k, v) for k, v in ctx._asdict().iteritems()]
                )
            ))
        
        
        cached = engine.get_cached((element, ctx.density_view))

        if not cached is None:
            logging.debug("Engine: Cache hit for {}: {}".format(element, cached))
            return cached
        else:
            if self._verbose:
                logging.debug("Can't find in the cache: \n elem: {},\n ctx: \n\t{}".format(
                    element, 
                    "\n\t".join(
                        ["{} -> {}".format(k, v) for k, v in ctx._asdict().iteritems()]
                    )
                ))


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
        
        if not isinstance(element, Variable):
            engine.cache((element, ctx.density_view), result)
        return result

    def _default_callback(self, element, ctx, engine):
        args_result = []
        for elem in element.get_args():
            args_result.append(self._deduce(elem, ctx, engine))

        return engine(element, Engine.make_ctx(tuple(args_result)))

    def _function_result(self, element, ctx, engine):
        if isinstance(element.get_fun(), Function):
            elem_structure = self._structure.get(element.get_fun(), ctx.requested_shape)
            assert not elem_structure is None, "Need to provide structure information for {}".format(element)

            args_result = []
            for arg_elem in element.get_args():
                args_result.append(
                    self._deduce(
                        arg_elem, 
                        Parser.get_ctx_with(
                            ctx, 
                            requested_shape=elem_structure
                        ), 
                        engine
                    )
                )
            
            return engine(
                element.get_fun(),
                Engine.make_ctx(tuple(args_result), elem_structure, ctx.scope)
            )
        else:
            args_result = []
            for elem in element.get_args():
                args_result.append(self._deduce(elem, ctx, engine))
            return engine(element.get_fun(), Engine.make_ctx(tuple(args_result)))

    def _deduce_variable_density(self, variable, density, ctx, engine):
        elem_structure = self._structure.get(variable)
        assert not elem_structure is None, "Need to provide structure information for {}".format(variable)

        var_dict = OrderedDict()
        arg_result = self._deduce(
            density, 
            Parser.get_ctx_with(
                ctx,
                variables_tree=var_dict,
                requested_shape=elem_structure
            ), 
            engine
        )
        ctx.variables_tree[variable] = var_dict
        return engine(variable, Engine.make_ctx((arg_result,), elem_structure))

    def _probability(self, element, ctx, engine):
        density = element.get_density()
        assert not density is None, "Can't find specification of {}".format(element)
        
        density_view = ctx.density_view if ctx.density_view == Density.View.DENSITY else Density.View.PROBABILITY 
        assert density_view != Density.View.PROBABILITY or element.is_log_form(), \
            "To deduce likelihood probability must be uplifted with the log function. Use log({})".format(element)

        return self._deduce_variable_density(
            element.get_output(), 
            density,
            Parser.get_ctx_with(
                ctx, 
                statement_id=element.get_statement_id(), 
                dependencies=element.get_dependencies(),
                scope=element.get_scope_name(),
                density_view=density_view,
                probability_output=element.get_output(),
            ), 
            engine
        )

    def _density(self, element, ctx, engine):
        args_result = []
        for elem in element.get_args():
            args_result.append(
                self._deduce(
                    elem, 
                    Parser.get_ctx_with(
                        ctx,
                        density_view=Density.View.SAMPLE,
                    ),
                    engine,
                )
            )

        if ctx.density_view == Density.View.SAMPLE:
            assert not ctx.requested_shape is None, "Shape information is not provided to sample {}".format(element)
            logging.debug("Sampling {} with shape {}x{}".format(element, self._batch_size, ctx.requested_shape))

            return engine(element, Engine.make_ctx(
                tuple(args_result), 
                structure=(self._batch_size, ctx.requested_shape),
                density_view=ctx.density_view,
            ))
        elif ctx.density_view == Density.View.PROBABILITY:
            provided_input = None
            if isinstance(ctx.probability_output, PartOfSequence) and not ctx.sequence_variables is None:
                logging.debug("Got likelihood of part of sequence {}, will try to find data from input to RNN".format(ctx.probability_output))
                assert ctx.probability_output in ctx.sequence_variables, "Couldn't find sequence data in sequence info for {}".format(ctx.output)
                provided_input = ctx.sequence_variables[ctx.probability_output]

            var_structure = self._structure.get(ctx.probability_output)
            assert not var_structure is None, "Need to provide structure information for `{}'".format(element)

            return engine(element, Engine.make_ctx(
                tuple(args_result), 
                density_view=ctx.density_view,
                structure=(self._batch_size, var_structure),
                provided_input=provided_input,
                input_variable=ctx.probability_output,
            ))
        elif ctx.density_view == Density.View.DENSITY:
            return engine(element, Engine.make_ctx(
                tuple(args_result), 
                density_view=ctx.density_view,
            ))
        else:
            raise Exception("Unexpected density view")

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

        var_structure = self._structure.get(element)
        assert not var_structure is None, "Need to provide structure information for `{}'".format(element)

        if not next_density is None:
            return self._deduce_variable_density(
                element, 
                next_density,
                Parser.get_ctx_with(
                    ctx, 
                    statement_id=next_probability.get_statement_id(), 
                    dependencies=next_probability.get_dependencies(),
                    scope=next_probability.get_scope_name(),
                ), 
                engine
            )
        ctx.variables_tree[element] = None
        return engine(element, Engine.make_ctx(structure=(self._batch_size, var_structure)))
        
    def _part_of_sequence(self, element, ctx, engine):
        idx = element.get_idx()
        if idx == 0:
            return self._variable(element, ctx, engine)
        elif idx == -1:
            t_elem = get_zero_offset(element)
            ret = self._deduce(t_elem, ctx, engine)
            return engine(element, Engine.make_ctx(ret))
        elif isinstance(idx, Index):
            seq = element.get_seq()
            if idx.get_offset() == -1: # t-1
                seq_init = self._deduce(seq[0], ctx, engine)
                return engine(element, Engine.make_ctx(seq_init))
            elif idx.get_offset() == 0: # t
                seq_t = self._variable(element, ctx, engine)
                return seq_t
            else:
                raise Exception("Unsupported index offset: {}".format(idx))
        else:
            raise Exception("Unsupported indexing {}".format(idx))

    def _metrics(self, element, ctx, engine):
        args_result = []
        for elem in element.get_args():
            args_result.append(
                self._deduce(
                    elem, 
                    Parser.get_ctx_with(
                        ctx, 
                        density_view=Density.View.DENSITY
                    ), 
                   engine
                )
            )
        return engine(element, Engine.make_ctx(tuple(args_result)))

