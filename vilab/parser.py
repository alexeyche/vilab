

from vilab.api import *
from vilab.util import is_sequence
from vilab.log import setup_log

import logging
import copy
import numbers
import types
from collections import namedtuple, OrderedDict
import numpy as np
import gc
import os

class DensityView(object):
    SAMPLE = "sample"
    PROBABILITY = "probability"
    DENSITY = "density"


class Parser(object):
    Ctx = namedtuple("Ctx", [
        "statement_id", 
        "output", 
        "requested_shape", 
        "probability", 
        "dependencies", 
        "density_view",
        "is_part_of_sequence",
        "sequence_info"
    ])

    @staticmethod
    def get_ctx_with(ctx, 
        statement_id=None, 
        output=None, 
        requested_shape=None, 
        probability=None, 
        dependencies=None, 
        density_view=None, 
        is_part_of_sequence=None,
        sequence_info=None
    ):
        return Parser.Ctx(
            ctx.statement_id if statement_id is None else statement_id,
            ctx.output if output is None else output,
            ctx.requested_shape if requested_shape is None else requested_shape,
            ctx.probability if probability is None else probability,
            ctx.dependencies if dependencies is None else dependencies,
            ctx.density_view if density_view is None else density_view,
            ctx.is_part_of_sequence if is_part_of_sequence is None else is_part_of_sequence,
            ctx.sequence_info if sequence_info is None else sequence_info,
        )

    def __init__(self, engine, output, feed_dict, structure, batch_size, reuse=False, context=None):
        self.engine = engine
        self.feed_dict = feed_dict
        self.structure = structure
        self.batch_size = batch_size
        self.reuse = reuse
        probability = None
        if context:
            if isinstance(context, Probability):
                probability = context
            else:
                raise Exception("Unsupported type of context: {}".format(context))
        self.default_ctx = Parser.Ctx(None, output, None, probability, None, DensityView.SAMPLE, False, None)

        self.variable_set = set()
        self.statements_met = set()
        
        self.visited = {}
        self.engine_inputs = {}
        
        self.shape_info = []

        self.sample_configuration_stack = []

        self.type_callbacks = {
            Probability: self.probability,
            Density: self.density,
            Variable: self.variable,
            FunctionResult: self.function_result,
            Metrics: self.metrics,
            numbers.Integral: self.identity,
            numbers.Real: self.identity,
            SequenceOperation: self.sequence_operation,
        }
        self.level = 0


    def get_engine_inputs(self):
        return self.engine_inputs

    def get_ctx(self):
        return self.default_ctx

    def get_visited_value(self, elem, ctx):
        if isinstance(elem, Variable):
            return None
        key = (elem, ctx.density_view)
        if key in self.visited and not elem in self.feed_dict:
            logging.debug("CACHE HIT: Element {} is already visited. Returning {} and ignoring ...".format(key, self.visited[key]))
            return self.visited[key]
        # logging.debug(": {} is not found, calculating".format(key))
        return None

    def update_visited_value(self, elem, ctx, value):
        if isinstance(elem, Variable):
            return
        key = (elem, ctx.density_view)
        if not key in self.visited:
            self.visited[key] = value
            logging.debug("Saving {} -> {}".format(key, value))

    def deduce(self, elem, ctx = None):
        if ctx is None:
            ctx = self.default_ctx

        visited_value = self.get_visited_value(elem, ctx)
        if not visited_value is None:
            return visited_value
        
        logging.debug("level: {}, elem: {}, ctx: {}".format(self.level, elem, ctx))
        if logging.getLogger().level == logging.DEBUG:
            setup_log(logging.DEBUG, ident_level=self.level)        
        
        cb_to_call = [ cb for tp, cb in self.type_callbacks.iteritems() if isinstance(elem, tp)]

        assert len(cb_to_call) > 0, "Deducer got unexpected element: {}".format(elem)
        assert len(cb_to_call) == 1, "Got too many callback matches for element: {}".format(elem)
        self.level += 1
        
        result = cb_to_call[0](elem, ctx)
        
        # self.shape_info.append(self.engine.get_shape(result))
        
        self.update_visited_value(elem, ctx, result)
        
        self.level -= 1
        if logging.getLogger().level == logging.DEBUG:
            setup_log(logging.DEBUG, ident_level=self.level)        
        
        logging.debug("level out: {}, result: {}".format(self.level, result))
        return result


    def probability(self, prob, ctx):
        logging.debug("Deducer({}): deducing for probability".format(prob))
        
        model, output, dependencies = prob.get_components()
        
        assert model.has_description(output, dependencies), \
            "Couldn't find description of variable {} condition on dependencies {} for model {}".format(output, dependencies, model)
        
        density_view = ctx.density_view if ctx.density_view == DensityView.DENSITY else DensityView.PROBABILITY 
        
        assert density_view != DensityView.PROBABILITY or prob.is_log_form(), \
            "To deduce likelihood probability must be uplifted with the log function. Use log({})".format(prob)

        return self.deduce(
            output, 
            Parser.get_ctx_with(
                ctx,
                output=output, 
                probability=prob,
                dependencies=dependencies,
                density_view=density_view
            )
        ) 


    def density(self, elem, ctx):
        cfg = elem.get_config()

        probability_with_deps = None
        if not ctx.probability is None and ctx.probability.get_dependencies() is None:
            probability_with_deps = self.deduce_dependencies_for_probability(elem, ctx.probability)

        logging.debug("Deducer({}): deducing for density".format(elem))
        dst_elem = copy.copy(elem)
        logging.debug("Deducer({}): Deducing arguments ...".format(elem))
        dst_elem.set_args([
            self.deduce(
                arg, 
                Parser.get_ctx_with(
                    ctx, 
                    density_view=DensityView.SAMPLE,
                    dependencies=ctx.dependencies if probability_with_deps is None else probability_with_deps.get_dependencies(),
                    probability=ctx.probability if probability_with_deps is None else probability_with_deps
                ))
            for arg in elem.get_args()
        ])
        logging.debug("Deducer({}): Done".format(elem))

        if ctx.density_view == DensityView.SAMPLE:
            assert not ctx.requested_shape is None, "Shape information is not provided to sample {}".format(dst_elem)
            logging.info("Sampling {} with shape {}x{}".format(elem, self.batch_size, ctx.requested_shape))
        
            self.sample_configuration_stack.append(cfg)
            
            return self.engine.sample(
                dst_elem, 
                (self.batch_size, ctx.requested_shape), 
                importance_samples=cfg.importance_samples
            )
        elif ctx.density_view == DensityView.PROBABILITY:
            assert ctx.output in self.feed_dict, "Expecting {} in feed dictionary to calculate probability using PDF {}".format(ctx.output, elem)
            assert not isinstance(dst_elem, DiracDelta), "Can't calculate likelihood value for DiracDelta distribution"

            data = self.feed_dict[ctx.output]
            provided_input = self.engine.provide_input(elem.get_name(), (self.batch_size, ) + data.shape[1:])

            assert not provided_input in self.engine_inputs, "Visiting input for {} again".format(elem.get_name())
            self.engine_inputs[provided_input] = ctx.output

            logging.info("Deducing likelihood for {} provided from inputs".format(ctx.output))

            return self.engine.likelihood(dst_elem, provided_input)
        elif ctx.density_view == DensityView.DENSITY:
            logging.info("Return density parameters for {}".format(dst_elem))
            return self.engine.get_density(dst_elem)


    def deduce_variable(self, elem, ctx):
        next_statement_id, next_density, next_probability = None, None, None

        if ctx.probability is None or elem != ctx.probability.get_output():
            candidates = []

            for m in elem.get_models():
                for statement_id, density, probability in m.get_var_probabilities(elem):
                    min_global_statement = ctx.statement_id is None or statement_id < ctx.statement_id  # intersected with current statement id
                    
                    if min_global_statement:
                        candidates.append((statement_id, density, probability))

            if len(candidates) == 1:
                logging.debug("Got 1 candidate {} to describe {}. Easy choice".format(candidates[0], elem))
                next_statement_id, next_density, next_probability = candidates[0]
            
            elif len(candidates) > 1:
                logging.debug("Got {} candidates to describe variable {}, need to choose ...".format(len(candidates), elem))
                assert not ctx.probability is None, "Got too many descriptions of variable {}: {}; failing to choose one of them (need more context)" \
                    .format(elem, ", ".join([str(c[2]) for c in candidates]))

                if not ctx.probability.get_dependencies() is None: 
                    deps = set([d for d in ctx.probability.get_dependencies() if d != elem])
                    if len(deps) == 0:
                        logging.debug("Looking for description that is unconditioned")

                        candidates = [ (c_st, c_dens, c_prob) for c_st, c_dens, c_prob in candidates if len(c_prob.get_dependencies()) == 0]
                        
                        assert len(candidates) > 0, "Failed to find any description of {} which includes unconditioned dependency".format(elem)
                        assert len(candidates) == 1, "Got too many descriptions of variable {} which includes unconditioned dependency".format(elem)                    
                    else:
                        logging.debug("Looking for description that has {} as subset".format(deps))
                        
                        candidates = [ (c_st, c_dens, c_prob) for c_st, c_dens, c_prob in candidates if deps <= set(c_prob.get_dependencies())]
                    
                        assert len(candidates) > 0, "Failed to find any description of {} which has dependencies that includes {}".format(elem, deps)
                        assert len(candidates) == 1, "Got too many descriptions of variable {} which has dependencies that includes {}".format(elem, deps)
                else:
                    raise Exception("Got variable with unknown dependencies. Alexey, please, fix this shit")

                logging.debug("Found this one: {}".format(candidates[0][2]))
                
                next_statement_id, next_density, next_probability = candidates[0]
        else:
            for m in elem.get_models():
                for statement_id, density, probability in m.get_var_probabilities(elem):
                    if ctx.probability == probability:
                        assert next_statement_id is None, "Got duplicated probability"
                        next_statement_id, next_density, next_probability = statement_id, density, probability

            assert not next_statement_id is None, "Failed to find how variable {} is described by probability {}".format(elem, probability)
        return next_statement_id, next_density, next_probability
        

    def variable(self, elem, ctx):
        is_part_of_sequence = isinstance(elem, PartOfSequence)

        logging.debug("Deducer({}): deducing for variable".format(elem))

        recursion = elem in self.variable_set
        if recursion:
            if not elem in self.feed_dict:
                raise Exception("Expecting data on the top for {}".format(elem))
            else:
                logging.info("Deducer reached to the top of {}, taking value from inputs".format(elem))
        else:
            self.variable_set.add(elem)

        next_statement_id, next_density, next_probability = self.deduce_variable(elem, ctx)

        assert not next_statement_id is None or elem in self.feed_dict or is_part_of_sequence, "Failed to deduce variable {}, need to provide data for variable".format(elem)
        
        if next_statement_id is None and is_part_of_sequence:
            assert not ctx.sequence_info is None, "Got part of sequence without sequence_info"
            assert elem in ctx.sequence_info, "Couldn't find sequence data in sequence info for {}".format(elem)
            return ctx.sequence_info[elem]
            

        if next_statement_id is None and elem in self.feed_dict:
            logging.debug("Deducer({}): Found variable in inputs".format(elem))
        
            data = self.feed_dict[elem]
            provided_input = self.engine.provide_input(elem.get_scope_name(), (self.batch_size, ) + data.shape[1:])
            assert not provided_input in self.engine_inputs, "Visiting input for {} again".format(elem.get_name())
            self.engine_inputs[provided_input] = elem
            if not recursion:
                self.variable_set.remove(elem)
            return provided_input
        
        cache_key = (elem, next_probability, ctx.density_view, next_statement_id)
        if cache_key in self.visited:
            logging.debug("CACHE HIT, Variable {}, stat.id #{}: ".format(elem, next_statement_id))
            return self.visited[cache_key]

        elem_structure = self.structure.get(elem)
        assert not elem_structure is None, "Need to provide structure information for {}".format(elem)
        
        logging.debug("Deducer({}): deducing density {}".format(elem, next_density))
        
        logging.info("Deducing variable {}, statement #{}".format(elem.get_name(), next_statement_id))
        result = self.deduce(
            next_density, 
            Parser.get_ctx_with(
                ctx,
                statement_id=next_statement_id,
                output=next_probability.get_output(),
                requested_shape=elem_structure, 
                probability=next_probability, 
                density_view=ctx.density_view,
                dependencies=next_probability.get_dependencies(),
                is_part_of_sequence=is_part_of_sequence
            )
        )
        
        self.visited[cache_key] = result

        if not recursion:
            self.variable_set.remove(elem)
        return result

    def get_variables(self, elem):
        vars_met = set()
        if isinstance(elem, Variable):
            vars_met.add(elem)
        elif isinstance(elem, FunctionResult) or isinstance(elem, Density):
            for v in elem.get_args():
                vars_met |= self.get_variables(v)
        else:
            raise Exception("Failed to find variables in branch, unknown type: {}".format(elem))
        return vars_met

    def deduce_dependencies_for_probability(self, elem, prob):
        logging.debug("Got unknown dependencies, need to try to collect this information ...")
        var_deps = set()
        for a in elem.get_args():
            var_deps |= self.get_variables(a)
        
        logging.debug("For the function {} collected dependencies: {}".format(elem, var_deps))
        return Probability(prob.get_model(), prob.get_output(), var_deps)


    def function_result(self, elem, ctx):
        logging.debug("Deducer({}): deducing for function result".format(elem))
        
        probability_with_deps = None
        if not ctx.probability is None and ctx.probability.get_dependencies() is None:
            probability_with_deps = self.deduce_dependencies_for_probability(elem, ctx.probability)

        deduced_args = [
            self.deduce(
                arg, 
                Parser.get_ctx_with(
                    ctx, 
                    density_view=DensityView.SAMPLE,
                    dependencies=ctx.dependencies if probability_with_deps is None else probability_with_deps.get_dependencies(),
                    probability=ctx.probability if probability_with_deps is None else probability_with_deps
                ))
            for arg in elem.get_args()
        ]

        if isinstance(elem.get_fun(), Function):
            elem_structure = self.structure.get(elem.get_fun(), ctx.requested_shape)
            
            assert not elem_structure is None, "Need to provide structure information for {}".format(elem)
            
            context_name = ""
            if not ctx.probability is None:
                context_name = ctx.probability.get_context_name()

            logging.info("Calling function {}{} with structure {}, act: {}, arguments: {}".format(
                "" if ctx.probability is None else str(ctx.probability) + "/", elem.get_name(), elem_structure, elem.get_act().get_name() if elem.get_act() else "linear", ",".join([str(a.get_name()) for a in elem.get_args()])
            ))
            
            cfg = elem.get_fun().get_config()

            return self.engine.function(
                *deduced_args, 
                size = elem_structure, 
                name = elem.get_name(),
                scope_name = context_name,
                act = elem.get_act(),
                reuse = self.reuse,
                weight_factor = cfg.weight_factor,
                use_batch_norm = cfg.use_batch_norm if not elem.get_act() is None else False
            )
        elif isinstance(elem.get_fun(), BasicFunction):
            logging.info("Calling basic function {}, arguments: {}".format(
                elem.get_name(), ",".join([str(a.get_name()) if hasattr(a, "get_name") else str(a) for a in elem.get_args()])
            ))
            return self.engine.calc_basic_function(elem.get_fun(), *deduced_args)
        else:
            raise Exception("Unknown type of function: {}".format(elem.get_fun()))


    def metrics(self, elem, ctx):
        logging.debug("Deducer({}): deducing for metrics".format(elem))

        deduced_args = [
            self.deduce(arg, Parser.get_ctx_with(ctx, density_view=DensityView.DENSITY)) 
            for arg in elem.get_args()
        ]
        
        logging.info("Calculating metrics {} with arguments {}".format(
            elem.get_name(),", ".join([str(a.get_name()) if hasattr(a, "get_name") else str(a) for a in elem.get_args()])
        ))
        
        return self.engine.calculate_metrics(elem, *deduced_args)


    def identity(self, elem, ctx):
        return elem

    def deduce_sequence(self, elem, ctx):
        variables = set([ item for v in Sequence.REGISTER.values() for item in v.get_parts().values() ])
        
        state_variables, input_variables = [], []

        for v in variables:
            if isinstance(v, PartOfSequence):
                idx = v.get_idx()
                
                if not isinstance(idx, Index):
                    continue
                
                if idx.get_offset() == 0: # input data
                    input_variables.append(v)
                elif idx.get_offset() == -1: # state data
                    state_variables.append(v)
                else:
                    raise Exception("Index offset that is not 0 or -1 is not supported yet, got {}".format(idx.get_offset()))
        
        output_state_variables = []
        for state_var in state_variables:
            seq = state_var.get_seq()
            seq_parts = seq.get_parts()
            next_idx = state_var.get_idx() + 1
            assert next_idx in seq_parts, "Need to define generation process for sequence {} (define {}[{}])".format(
                seq, seq, next_idx
            )
            output_state_variables.append(seq_parts[next_idx])
        
        for state_var in output_state_variables:
            if state_var in input_variables:
                input_variables.remove(state_var)


        logging.debug("Preparing to run rnn with input variables: {}, state variables: {}".format(
            ", ".join([str(v) for v in input_variables]),
            ", ".join([str(v) for v in state_variables]),
        ))
        
        logging.debug("And with output state variables: {}".format(
            ", ".join([str(v) for v in output_state_variables]),
        )) 

        def get_value(input_tuple, state_tuple):
            for var, data in zip(state_variables, state_tuple):
                sequence_info[var] = data
            for var, data in zip(input_variables, input_tuple):
                sequence_info[var] = data
            
            res = self.deduce(
                elem,
                ctx = Parser.get_ctx_with(
                    ctx,
                    sequence_info=sequence_info,
                )
            )

            output_state = []
            for state_var in output_state_variables:
                output_state.append(
                    self.deduce(
                        state_var,
                        ctx = Parser.get_ctx_with(
                            ctx,
                            sequence_info=sequence_info,
                        )
                    )
                )

            return res, tuple(output_state)

        self.engine.run_sequence(get_value)

    def sequence_operation(self, elem, ctx):
        logging.debug("Got sequence operation: {}".format(elem))
        self.deduce_sequence(elem.get_expr(), ctx)
        self.engine.get_sequence()
        return 

