

from engine import Engine
from api import *
from util import is_sequence
from log import setup_log

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
    Ctx = namedtuple("Ctx", ["output", "requested_shape", "probability", "dependencies", "density_view"])

    @staticmethod
    def get_ctx_with(ctx, output=None, requested_shape=None, probability=None, dependencies=None, density_view=None):
        return Parser.Ctx(
            ctx.output if output is None else output,
            ctx.requested_shape if requested_shape is None else requested_shape,
            ctx.probability if probability is None else probability,
            ctx.dependencies if dependencies is None else dependencies,
            ctx.density_view if density_view is None else density_view
        )

    def __init__(self, output, feed_dict, structure, batch_size, reuse=False, context=None):
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
        self.default_ctx = Parser.Ctx(output, None, probability, None, DensityView.SAMPLE)

        self.variable_set = set()
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
        }
        self.level = 0


    def get_engine_inputs(self):
        return self.engine_inputs

    def get_ctx(self):
        return self.default_ctx

    def get_visited_value(self, elem, ctx):
        key = (elem, ctx.density_view)
        if key in self.visited and not elem in self.feed_dict:
            logging.debug("CACHE HIT: Element {} is already visited. Returning {} and ignoring ...".format(key, self.visited[key]))
            return self.visited[key]
        # logging.debug(": {} is not found, calculating".format(key))
        return None

    def update_visited_value(self, elem, ctx, value):
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
        
        self.shape_info.append(Engine.get_shape(result))
        
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
            Parser.Ctx(
                output, 
                ctx.requested_shape,
                prob,
                dependencies,
                density_view
            )
        ) 


    def density(self, elem, ctx):
        cfg = elem.get_config()

        logging.debug("Deducer({}): deducing for density".format(elem))
        dst_elem = copy.copy(elem)
        logging.debug("Deducer({}): Deducing arguments ...".format(elem))
        dst_elem.set_args([
            self.deduce(
                arg, 
                Parser.get_ctx_with(
                    ctx, 
                    density_view=DensityView.SAMPLE)
                ) 
            for arg in elem.get_args()
        ])
        logging.debug("Deducer({}): Done".format(elem))

        if ctx.density_view == DensityView.SAMPLE:
            assert not ctx.requested_shape is None, "Shape information is not provided to sample {}".format(dst_elem)
            logging.info("Sampling {} with shape {}x{}".format(elem, self.batch_size, ctx.requested_shape))
        
            self.sample_configuration_stack.append(cfg)
            
            return Engine.sample(
                dst_elem, 
                (self.batch_size, ctx.requested_shape), 
                importance_samples=cfg.importance_samples
            )
        elif ctx.density_view == DensityView.PROBABILITY:
            assert ctx.output in self.feed_dict, "Expecting {} in feed dictionary to calculate probability using PDF {}".format(ctx.output, elem)
            assert not isinstance(dst_elem, DiracDelta), "Can't calculate likelihood value for DiracDelta distribution"

            data = self.feed_dict[ctx.output]
            provided_input = Engine.provide_input(elem.get_name(), (self.batch_size, ) + data.shape[1:])

            assert not provided_input in self.engine_inputs, "Visiting input for {} again".format(elem.get_name())
            self.engine_inputs[provided_input] = ctx.output

            logging.info("Deducing likelihood for {} provided from inputs".format(ctx.output))

            return Engine.likelihood(dst_elem, provided_input)
        elif ctx.density_view == DensityView.DENSITY:
            logging.info("Return density parameters for {}".format(dst_elem))
            return Engine.get_density(dst_elem)


    def variable(self, elem, ctx):
        logging.debug("Deducer({}): deducing for variable".format(elem))

        trivial_deduce = not ctx.output is None and \
                         isinstance(ctx.output, Variable) and \
                         elem == ctx.output and \
                         elem in self.feed_dict and \
                         not elem in self.variable_set
        
        recursion = elem in self.variable_set
        if recursion:
            if not elem in self.feed_dict:
                raise Exception("Expecting data on the top for {}".format(elem))
            else:
                logging.info("Deducer reached to the top of {}, taking value from inputs".format(elem))
        else:
            self.variable_set.add(elem)
        
        
        if trivial_deduce:
            logging.debug("Found output variable {} in inputs, considering this as trivial deduce, will proceed graph calculation".format(elem))
        
        assert ctx.output == elem or ctx.dependencies is None or elem in ctx.dependencies, \
            "Deducer met variable {} that wasn't specified as dependent variable".format(elem) 
        
        if elem in self.feed_dict and not trivial_deduce:
            logging.debug("Deducer({}): Found variable in inputs".format(elem))
        
            data = self.feed_dict[elem]
            provided_input = Engine.provide_input(elem.get_name(), (self.batch_size, ) + data.shape[1:])
            assert not provided_input in self.engine_inputs, "Visiting input for {} again".format(elem.get_name())
            self.engine_inputs[provided_input] = elem
            if not recursion:
                self.variable_set.remove(elem)
            return provided_input
        else:
            if ctx.probability is None: # trying to deduce things for just variable
                logging.debug("Found no probability in context, trying to do our best ...")
                elem_models = elem.get_models()
                assert len(elem_models) > 0, "Variable {} is not described by any probability density, can't deduce".format(elem)
                assert len(elem_models) == 1, "Variable {} is described by many probability densities, need to define proper context".format(elem)
                elem_model = elem_models[0]
                
                dst_probs_and_dens = elem_model.get_var_probabilities(elem)
                assert len(dst_probs_and_dens) > 0, "Couldn't find any description of variable {}".format(elem)
                assert len(dst_probs_and_dens) == 1, "Need one description, FIX ME"
                probability, dst_density = dst_probs_and_dens[0]
            else:
                if ctx.probability.get_output() != elem:
                    logging.debug("Got variable jumping case {} -> {}".format(ctx.probability.get_output(), elem))
                    models = elem.get_models()
                    assert len(models) == 1, "Need one model, FIX ME" 
                    elem_model = models[0]

                    dst_probs_and_dens = elem_model.get_var_probabilities(elem)
                    assert len(dst_probs_and_dens) == 1, "Need one description, FIX ME"
                    probability, dst_density = dst_probs_and_dens[0]
                else:
                    probability = ctx.probability
                    dst_density = ctx.probability.get_density()

            elem_model, _, deps = probability.get_components()
            
            elem_structure = self.structure.get(elem)
            assert not elem_structure is None, "Need to provide structure information for {}".format(elem)
            
            logging.debug("Deducer({}): deducing density {}".format(elem, dst_density))
            
            logging.info("Deducing variable {}".format(elem.get_name()))
            result = self.deduce(
                dst_density, 
                Parser.get_ctx_with(
                    ctx, 
                    requested_shape=elem_structure, 
                    probability=probability, 
                    density_view=ctx.density_view,
                    dependencies = deps
                )
            )
            
            if not recursion:
                self.variable_set.remove(elem)
            return result


    def function_result(self, elem, ctx):
        logging.debug("Deducer({}): deducing for function result".format(elem))
        
        deduced_args = [
            self.deduce(arg, Parser.get_ctx_with(ctx, density_view=DensityView.SAMPLE))
            for arg in elem.get_args()
        ]

        if isinstance(elem.get_fun(), Function):
            elem_structure = self.structure.get(elem.get_fun(), ctx.requested_shape)
            
            assert not elem_structure is None, "Need to provide structure information for {}".format(elem)
            
            context_name = ""
            if not ctx.probability is None:
                context_name = ctx.probability.get_context_name()


            logging.info("Calling function {}{} with structure {}, act: {}, arguments: {}".format(
                context_name, elem.get_name(), elem_structure, elem.get_act().get_name() if elem.get_act() else "linear", ",".join([str(a.get_name()) for a in elem.get_args()])
            ))
            
            cfg = elem.get_fun().get_config()

            return Engine.function(
                *deduced_args, 
                size = elem_structure, 
                name = "{}{}".format(context_name, elem.get_name()),
                act = elem.get_act(),
                reuse = self.reuse,
                weight_factor = cfg.weight_factor,
                use_batch_norm = cfg.use_batch_norm if not elem.get_act() is None else False
            )
        elif isinstance(elem.get_fun(), BasicFunction):
            logging.info("Calling basic function {}, arguments: {}".format(
                elem.get_name(), ",".join([str(a.get_name()) if hasattr(a, "get_name") else str(a) for a in elem.get_args()])
            ))
            return Engine.calc_basic_function(elem.get_fun(), *deduced_args)
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
        
        return Engine.calculate_metrics(elem, *deduced_args)


    def identity(self, elem, ctx):
        return elem


def get_data_size(feed_dict):
    batch_size = None
    for k, v in feed_dict.iteritems():
        if not batch_size is None:
            assert batch_size == v.shape[0], "Batch size is not the same through feed data"
        else:
            batch_size = v.shape[0]    
    return batch_size


def deduce_shapes(feed_dict, structure):
    for k, v in feed_dict.iteritems():
        if not k in structure:
             structure[k] = v.shape[-1]


def get_data_slice(element_id, batch_size, feed_dict):
    data_slice = {}
    for k, v in feed_dict.iteritems():
        v_shape = v.shape
        next_element_id = min(element_id + batch_size, v_shape[0])
        data_v = v[element_id:next_element_id]
        data_len = next_element_id - element_id
        if batch_size > data_len:
            data_v = np.concatenate([data_v, np.zeros((batch_size - data_len,) + v_shape[1:])])
        data_slice[k] = data_v
    
    return data_slice, element_id + batch_size


def _run(leaves_to_run, feed_dict, batch_size, engine_inputs):
    element_id = 0
    outputs = None
    while len(feed_dict) == 0 or element_id < get_data_size(feed_dict):
        data_for_input = {}
        data_slice, element_id = get_data_slice(element_id, batch_size, feed_dict)
        
        for k, v in engine_inputs.iteritems():
            data = data_slice.get(v)
            assert not data is None, "Failed to find {} in feed_dict".format(v)
            data_for_input[k] = data
        
        batch_outputs = Engine.run(
            leaves_to_run + Engine._debug_outputs, 
            data_for_input
        )
        if outputs is None:
            outputs = batch_outputs
        else:
            for out_id, (out, bout) in enumerate(zip(outputs, batch_outputs)):
                if not bout is None:
                    outputs[out_id] = np.concatenate([out, bout])

    return outputs  

def deduce(elem, feed_dict={}, structure={}, batch_size=None, reuse=False, silent=False, context=None):    
    log_level = logging.getLogger().level
    if silent:
        setup_log(logging.CRITICAL)        
        
    data_size = get_data_size(feed_dict)
    deduce_shapes(feed_dict, structure)
    
    assert not batch_size is None or data_size is None or data_size < 10000, \
        "Got too big data size, need to point proper batch_size in arguments"

    if batch_size is None:
        assert not data_size is None, "Need to specify batch size"
        batch_size = data_size

    results = []
    top_elem = elem
    if is_sequence(elem):
        top_elem = elem[0]
    
    if context:
        parser = Parser(context, feed_dict, structure, batch_size, reuse)
        parser.deduce(context)
    else:
        parser = Parser(top_elem, feed_dict, structure, batch_size, reuse)
        
    if is_sequence(elem):
        for subelem in elem:
            results.append(parser.deduce(subelem))
    else:
        results.append(parser.deduce(elem))

    logging.debug("Collected {}".format(results))
    outputs = _run(results, feed_dict, batch_size, parser.get_engine_inputs())

    if silent:
        setup_log(log_level)        

    if not is_sequence(elem):
        return outputs[0]
    return outputs

class Monitor(object):
    def __init__(self, elems, freq=1, feed_dict=None, callback=None):
        self.elems = elems
        self.freq= freq
        self.feed_dict = feed_dict
        self.callback = callback


def maximize(
    elem, 
    epochs=100, 
    learning_rate=1e-03, 
    feed_dict={}, 
    structure={}, 
    optimizer=Optimizer.ADAM, 
    batch_size=None, 
    monitor=Monitor([])
):
    data_size = get_data_size(feed_dict)
    deduce_shapes(feed_dict, structure)

    assert not batch_size is None or data_size < 10000 , "Got too big data size, need to point proper batch_size in arguments"

    if batch_size is None:
        batch_size = data_size
    
    parser = Parser(elem, feed_dict, structure, batch_size)
    to_optimize = parser.deduce(elem)
        
    opt_output = Engine.optimization_output(to_optimize, optimizer, learning_rate)

    to_monitor = OrderedDict()
    for m in monitor.elems:
        name = m.get_name() if hasattr(m, "get_name") else str(m)
        to_monitor[name] = parser.deduce(m, Parser.get_ctx_with(parser.get_ctx(), output=m))

    monitoring_values = []

    engine_inputs = parser.get_engine_inputs()

    logging.info("Optimizing provided value for {} epochs using {} optimizer".format(epochs, optimizer))
    for e in xrange(epochs):
        returns = _run([to_optimize, opt_output], feed_dict, batch_size, engine_inputs)

        if e % monitor.freq == 0:
            monitor_returns = _run(
                [to_optimize] + to_monitor.values(), 
                monitor.feed_dict if not monitor.feed_dict is None else feed_dict,
                batch_size, 
                parser.get_engine_inputs()
            )

            monitor_v = monitor_returns[1:]
            
            mon_str = ""
            if not monitor.feed_dict is None:
                mon_str = ", monitor value: {}".format(np.mean(monitor_returns[0]))

            logging.info("Epoch: {}, value: {}{}".format(e, np.mean(returns[0]), mon_str))
            if not monitor.callback is None:
                monitor.callback(e, *monitor_v)
            monitor_v = [np.mean(v) for v in monitor_v]
            for n, mv in zip(to_monitor.keys(), monitor_v):
                mon_str = "{}".format(np.mean(mv))
                    
                logging.info("    {}: {}".format(n, mon_str))
            
            monitoring_values.append(monitor_v)

        # _ = gc.collect()

    return returns[0], np.asarray(monitoring_values)
