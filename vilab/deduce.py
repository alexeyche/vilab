
from vilab.parser import Parser
import numpy as np

from vilab.api import *
from vilab.engines.tf_engine import TfEngine
from collections import OrderedDict

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
        if isinstance(k, Sequence):
            for _, part in k.get_parts().iteritems():
                if not part in structure:
                    structure[part] = v
            


def get_data_slice(element_id, batch_size, feed_dict):
    data_slice = {}
    for k, v in feed_dict.iteritems():
        v_shape = v.shape
        next_element_id = min(element_id + batch_size, get_batch_size(v))
        data_v = v[element_id:next_element_id]
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

    return outputs  

def deduce(elem, feed_dict={}, structure={}, batch_size=None, reuse=False, silent=False, context=None, engine=TfEngine()):
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
        parser = Parser(engine, context, feed_dict, structure, batch_size, reuse)
        parser.deduce(context)
    else:
        parser = Parser(engine, top_elem, feed_dict, structure, batch_size, reuse)
        
    if is_sequence(elem):
        for subelem in elem:
            results.append(parser.deduce(subelem))
    else:
        results.append(parser.deduce(elem))

    logging.debug("Collected {}".format(results))
    outputs = _run(engine, results, feed_dict, batch_size, parser.get_engine_inputs())

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
    monitor=Monitor([]),
    engine=TfEngine()
):
    data_size = get_data_size(feed_dict)
    deduce_shapes(feed_dict, structure)

    assert not batch_size is None or data_size < 10000 , "Got too big data size, need to point proper batch_size in arguments"

    if batch_size is None:
        batch_size = data_size
    
    parser = Parser(engine, elem, feed_dict, structure, batch_size)
    to_optimize = parser.deduce(elem)
        
    opt_output = engine.optimization_output(to_optimize, optimizer, learning_rate)

    to_monitor = OrderedDict()
    for m in monitor.elems:
        name = m.get_name() if hasattr(m, "get_name") else str(m)
        to_monitor[name] = parser.deduce(m, Parser.get_ctx_with(parser.get_ctx(), output=m))

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

    return returns[0], np.asarray(monitoring_values)
