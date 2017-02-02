

from engine import Engine
from api import *
from util import is_sequence

import logging
import copy
import numbers
import types
from collections import namedtuple
import numpy as np


class DensityView(object):
    SAMPLE = "sample"
    PROBABILITY = "probability"
    DENSITY = "density"


class Parser(object):
    Ctx = namedtuple("Ctx", ["output", "requested_shape", "model", "dependencies", "density_view"])

    @staticmethod
    def get_ctx_with(ctx, output=None, requested_shape=None, model=None, dependencies=None, density_view=None):
        return Parser.Ctx(
            ctx.output if output is None else output,
            ctx.requested_shape if requested_shape is None else requested_shape,
            ctx.model if model is None else model,
            ctx.dependencies if dependencies is None else dependencies,
            ctx.density_view if density_view is None else density_view
        )

    def __init__(self, output, feed_dict, structure, batch_size, reuse=False):
        self.feed_dict = feed_dict
        self.structure = structure
        self.batch_size = batch_size
        self.reuse = reuse
        self.default_ctx = Parser.Ctx(output, None, None, None, DensityView.SAMPLE)

        self.variable_set = set()
        self.visited = {}
        self.engine_inputs = {}

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
            logging.debug("Element {} is already visited. Returning {} and ignoring ...".format(key, self.visited[key]))
            return self.visited[key]
        logging.debug("{} is not found, calculating".format(key))
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
        
        cb_to_call = [ cb for tp, cb in self.type_callbacks.iteritems() if isinstance(elem, tp)]
        assert len(cb_to_call) > 0, "Deducer got unexpected element: {}".format(elem)
        assert len(cb_to_call) == 1, "Got too many callback matches for element: {}".format(elem)
        self.level += 1
        
        result = cb_to_call[0](elem, ctx)
        
        self.update_visited_value(elem, ctx, result)
        
        self.level -= 1
        logging.debug("level out: {}, result: {}".format(self.level, result))
        return result


    def probability(self, prob, ctx):
        logging.debug("Deducer({}): deducing for probability".format(prob))
        
        model, output, dependencies = prob.get_components()
        
        return self.deduce(
            output, 
            Parser.Ctx(
                output, 
                ctx.requested_shape,
                model,
                dependencies,
                ctx.density_view if ctx.density_view == DensityView.DENSITY else DensityView.PROBABILITY 
            )
        ) 


    def density(self, elem, ctx):
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
            return Engine.sample(dst_elem, (self.batch_size, ctx.requested_shape))
        elif ctx.density_view == DensityView.PROBABILITY:
            assert ctx.output in self.feed_dict, "Expecting {} in feed dictionary to calculate probability using PDF {}".format(ctx.output, elem)
            
            logging.info("Deducing likelihood for {} provided from inputs".format(ctx.output))
            return Engine.likelihood(dst_elem, self.feed_dict[ctx.output])
        elif ctx.density_view == DensityView.DENSITY:
            logging.info("Return density parameters for {}".format(dst_elem))
            return Engine.get_density(dst_elem)


    def variable(self, elem, ctx):
        logging.debug("Deducer({}): deducing for variable".format(elem))

        trivial_deduce = not ctx.output is None and \
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
        
        
        if elem in self.feed_dict and not trivial_deduce:
            logging.debug("Deducer({}): Found variable in inputs".format(elem))
        
            data = self.feed_dict[elem]
            provided_input = Engine.provide_input(elem.get_name(), data.shape)
            assert not provided_input in self.engine_inputs, "Visiting input for {} again".format(elem.get_name())
            self.engine_inputs[provided_input] = data
            if not recursion:
                self.variable_set.remove(elem)
            return provided_input
        else:
            elem_model = elem.get_model()
            elem_structure = self.structure.get(elem)
            assert not elem_structure is None, "Need to provide structure information for {}".format(elem)
            
            # for d in elem.get_dependencies():
            #     self.variable_set[elem].add(d)
            #     logging.debug("Deducer({}): deducing dependency {}".format(elem, d))
            #     r = self.deduce(
            #         d, 
            #         Parser.get_ctx_with(
            #             ctx, 
            #             model=elem_model, 
            #             density_view=DensityView.SAMPLE if ctx.model != elem_model else ctx.density_view
            #         )
            #     )
            #     logging.debug("Deducer({}): dependency result {}".format(elem, r))

            logging.debug("Deducer({}): deducing density {}".format(elem, elem.get_density()))
            
            logging.info("Deducing variable {}".format(elem.get_name()))
            result = self.deduce(
                elem.get_density(), 
                Parser.get_ctx_with(
                    ctx, 
                    requested_shape=elem_structure, 
                    model=elem_model, 
                    density_view=DensityView.SAMPLE if ctx.model != elem_model else ctx.density_view
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
                            
            logging.info("Calling function {}/{} with structure {}, arguments: {}".format(
                ctx.model.get_name(), elem.get_name(), elem_structure, ",".join([str(a.get_name()) for a in elem.get_args()])
            ))
            # requested_shape = elem.get_size()[0]
            return Engine.function(
                *deduced_args, 
                size = elem_structure, 
                name = "{}/{}".format(ctx.model.get_name(), elem.get_name()),
                act = elem.get_act(),
                reuse = self.reuse
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


def deduce_batch_size(feed_dict, batch_size):
    for k, v in feed_dict.iteritems():
        if not batch_size is None:
            assert batch_size == v.shape[0], "Batch size is not the same through feed data"
        else:
            batch_size = v.shape[0]
    
    assert not batch_size is None, "Need to specify batch size, system couldn't infer this from input data"
    return batch_size

def deduce(elem, feed_dict={}, structure={}, batch_size=None):    
    batch_size = deduce_batch_size(feed_dict, batch_size)
    results = []
    top_elem = elem
    if is_sequence(elem):
        top_elem = elem[0]
    
    parser = Parser(top_elem, feed_dict, structure, batch_size)
    if is_sequence(elem):
        for subelem in elem:
            results.append(parser.deduce(subelem))
    else:
        results.append(parser.deduce(elem))

    logging.debug("Collected {}".format(results))
    outputs = Engine.run(
        results, 
        parser.get_engine_inputs()
    )
    if is_sequence(elem):
        return outputs[0]
    return outputs



def maximize(
    elem, 
    epochs=100, 
    feed_dict={}, 
    structure={}, 
    optimizer=Optimizer.ADAM, 
    batch_size=None, 
    config={}, 
    monitor=[], 
    monitor_callback=None,
    monitor_freq=1
):
    batch_size = deduce_batch_size(feed_dict, batch_size)
    results = []
    
    # elem = -elem # minimization
    
    parser = Parser(elem, feed_dict, structure, batch_size)
    results.append(parser.deduce(elem))

    monitor_names = []
    for m in monitor:
        results.append(parser.deduce(m, Parser.get_ctx_with(parser.get_ctx(), output=m)))
        monitor_names.append(m.get_name() if hasattr(m, "get_name") else str(m))

    opt_output = Engine.optimization_output(results[0], optimizer, config)
    results.append(opt_output)

    logging.info("Optimizing provided value for {} epochs using {} optimizer".format(epochs, optimizer))
    for e in xrange(epochs):
        returns = Engine.run(
            results, 
            parser.get_engine_inputs()
        )
        result_v = returns[0]  
        monitor_v = returns[1:-1]
        if e % monitor_freq == 0:
            logging.info("Epoch: {}, value: {}, ".format(e, np.mean(result_v)))
            for n, v in zip(monitor_names, monitor_v):
                logging.info("\t{}: {}".format(n, np.mean(v)))
            
            if not monitor_callback is None:
                monitor_callback(e, *monitor_v)
    return result_v, returns[1:-1]