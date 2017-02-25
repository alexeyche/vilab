
import numpy as np
from collections import OrderedDict

from vilab.parser import Parser

from vilab.api import *
from vilab.engines.tf_engine import TfEngine
from vilab.engines.print_engine import PrintEngine
from vilab.log import setup_log


def get_batch_size(data):
    if len(data.shape) == 3:
        return data.shape[1]
    if len(data.shape) == 2:
        return data.shape[0]
    raise Exception("Unexpected data shape: {}".format(data))

def get_data_size(feed_dict):
    batch_size = None
    for k, v in feed_dict.iteritems():
        if not batch_size is None:
            assert batch_size == get_batch_size(v), "Batch size is not the same through feed data"
        else:
            batch_size = get_batch_size(v)
    return batch_size


def deduce_shapes(feed_dict, structure):
    for k, v in feed_dict.iteritems():
        if isinstance(k, Sequence):
            assert len(v.shape) == 3, "Input data for sequence must have alignment time x batch x dimension"

            for _, part in k.get_parts().iteritems():
                if not part in structure:
                    structure[part] = v.shape[-1]
        else:
            if not k in structure:
                 structure[k] = v.shape[-1]
    
    for k, v in structure.copy().iteritems():
        if isinstance(k, PartOfSequence):
            k = k.get_seq()
        if isinstance(k, Sequence):
            for _, part in k.get_parts().iteritems():
                if not part in structure:
                    structure[part] = v

def get_data_slice(element_id, batch_size, feed_dict):
    data_slice = {}
    for k, v in feed_dict.iteritems():
        v_shape = v.shape
        next_element_id = min(element_id + batch_size, get_batch_size(v))
        if len(v_shape) == 2:
            data_v = v[element_id:next_element_id, :]
        elif len(v_shape) == 3:
            data_v = v[:, element_id:next_element_id, :]
        data_len = next_element_id - element_id
        if batch_size > data_len:
            if len(v_shape) == 2:
                data_v = np.concatenate([data_v, np.zeros((batch_size - data_len, v_shape[1]))])
            elif len(v_shape) == 3:
                data_v = np.concatenate([data_v, np.zeros((v_shape[0], batch_size - data_len, v_shape[2]))])
        data_slice[k] = data_v
    return data_slice, element_id + batch_size


def _run(engine, leaves_to_run, feed_dict, batch_size, engine_inputs):
    element_id = 0
    outputs = None
    while len(feed_dict) == 0 or element_id < get_data_size(feed_dict):
        data_for_input = {}
        data_slice, element_id = get_data_slice(element_id, batch_size, feed_dict)
        
        for k, v in engine_inputs.iteritems():
            data = data_slice.get(v)
            assert not data is None, "Failed to find {} in feed_dict".format(v)
            data_for_input[k] = data
        
        batch_outputs = engine.run(
            leaves_to_run + engine._debug_outputs, 
            data_for_input
        )
        if outputs is None:
            outputs = batch_outputs
        else:
            for out_id, (out, bout) in enumerate(zip(outputs, batch_outputs)):
                if not bout is None:
                    outputs[out_id] = np.concatenate([out, bout])
        if not engine.is_data_engine(): # no need to slice data for this engine
            break
    return outputs  


class DeduceContext(object):
    def __init__(self, parser):
        self.parser = parser
        self.feed_dict = parser.data_info.get_feed_dict()
        self.structure = parser.structure
        self.batch_size = parser.batch_size
        self.engine = parser.engine

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "DeduceContext(\n\tstructure={},\n\tbatch_size={},\n\tfeed_dict keys={}\n)".format(self.structure, self.batch_size, self.feed_dict.keys())

def deduce(elem, feed_dict=None, structure=None, batch_size=None, reuse=False, silent=False, context=None, engine=TfEngine()):
    elem_is_sequence = is_sequence(elem)
    str_repr = None
    # if not isinstance(engine, PrintEngine) and context is None:
    #     str_repr, _ = deduce(elem, feed_dict, structure, batch_size, reuse, silent, context, engine=PrintEngine())

    log_level = logging.getLogger().level
    if silent:
        setup_log(logging.CRITICAL)        
    
    if not str_repr is None:
        logging.info("== DEDUCE ===============================")
        logging.info("String representation of deducing element:")
        logging.info("    {}".format(str_repr))
    
    parser = None
    if not context is None:
        feed_dict =  context.feed_dict if feed_dict is None else feed_dict
        structure = context.structure
        batch_size = context.batch_size
        parser = context.parser
        engine = context.engine
        parser.reuse = True

    data_size = get_data_size(feed_dict)
    deduce_shapes(feed_dict, structure)
    
    assert not batch_size is None or data_size is None or data_size < 10000, \
        "Got too big data size, need to point proper batch_size in arguments"

    if batch_size is None:
        assert not data_size is None, "Need to specify batch size"
        batch_size = data_size

    results = []
    if not is_sequence(elem):
        elem = [elem]
    
    if parser is None:
        parser = Parser(engine, elem[0], Parser.DataInfo(feed_dict), structure, batch_size, reuse)

    results = parser.do(elem)
    
    logging.debug("Collected {}".format(results))
    outputs = _run(engine, results, feed_dict, batch_size, parser.get_engine_inputs())

    if silent:
        setup_log(log_level)        

    if not elem_is_sequence:
        return outputs[0], DeduceContext(parser)
    return outputs, DeduceContext(parser)

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
    monitor=Monitor([]),
    engine=TfEngine()
):
    assert not isinstance(engine, PrintEngine), "Can't maximize with PrintEngine"

    # str_repr, _ = deduce(elem, feed_dict, structure, batch_size, reuse=False, silent=True, context=None, engine=PrintEngine())
    # logging.info("== MAXIMIZE ===============================")
    # logging.info("String representation of deducing element:\n\t\n{}".format(str_repr))
    
    data_size = get_data_size(feed_dict)
    deduce_shapes(feed_dict, structure)

    assert not batch_size is None or data_size < 10000 , "Got too big data size, need to point proper batch_size in arguments"

    if batch_size is None:
        batch_size = data_size
    
    parser = Parser(engine, elem, Parser.DataInfo(feed_dict), structure, batch_size)
    
    parser_result = parser.do([elem] + monitor.elems)
    
    to_optimize = parser_result[0]

    to_monitor = OrderedDict()
    for m, val in zip(monitor.elems, parser_result[1:]):
        name = m.get_name() if hasattr(m, "get_name") else str(m)
        to_monitor[name] = val

        
    opt_output = engine.optimization_output(to_optimize, optimizer, learning_rate)

    monitoring_values = []

    engine_inputs = parser.get_engine_inputs()

    logging.info("Optimizing provided value for {} epochs using {} optimizer".format(epochs, optimizer))
    for e in xrange(epochs):
        returns = _run(engine, [to_optimize, opt_output], feed_dict, batch_size, engine_inputs)

        if e % monitor.freq == 0:
            monitor_returns = _run(
                engine,
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

    return returns[0], np.asarray(monitoring_values), DeduceContext(parser)
