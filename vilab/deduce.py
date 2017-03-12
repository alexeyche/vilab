
import numpy as np
from collections import OrderedDict

from vilab.parser import Parser

from vilab.api import *
from vilab.engines.tf_engine import TfEngine
# from vilab.engines.print_engine import PrintEngine
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


def _run(engine, leaves_to_run, feed_dict, batch_size):
    element_id = 0
    outputs = None
    while len(feed_dict) == 0 or element_id < get_data_size(feed_dict):
        data_for_input = {}
        data_slice, element_id = get_data_slice(element_id, batch_size, feed_dict)
        
        batch_outputs = engine.run(
            leaves_to_run, 
            data_slice
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
    def __init__(self, engine, structure, batch_size, feed_dict):
        self.engine = engine
        self.structure = structure
        self.batch_size = batch_size
        self.feed_dict = feed_dict

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "DeduceContext(\n\tstructure={},\n\tbatch_size={},\n\tfeed_dict keys={}\n)".format(self.structure, self.batch_size, self.feed_dict.keys())


def _deduce_elements(elem, feed_dict, structure, batch_size, reuse, silent, context, engine):
    log_level = logging.getLogger().level
    if silent:
        setup_log(logging.CRITICAL)        

    if context is None:
        data_size = get_data_size(feed_dict)
    
        assert not batch_size is None or data_size is None or data_size < 10000, \
            "Got too big data size, need to point proper batch_size in arguments"

        if batch_size is None:
            assert not data_size is None, "Need to specify batch size"
            batch_size = data_size

    else:
        batch_size = context.batch_size
        structure = context.structure

    if not is_sequence(elem):
        elements = [elem]
    else:
        elements = elem

    deduce_shapes(feed_dict, structure)
        
    engine_to_run = engine
    if context is None: 
        context = DeduceContext(engine, structure, batch_size, feed_dict)
    else:
        engine_to_run = context.engine
    
    p = Parser(batch_size, structure)
    deduced = p.parse(elements, engine_to_run)
    
    return deduced, context

def deduce(elem, feed_dict=None, structure=None, batch_size=None, reuse=False, silent=False, context=None, engine=TfEngine()):
    deduced, context = _deduce_elements(elem, feed_dict, structure, batch_size, reuse, silent, context, engine)

    results = _run(
        context.engine, 
        deduced, 
        context.feed_dict if feed_dict is None else feed_dict, 
        context.batch_size if batch_size is None else context.batch_size
    )

    if not is_sequence(elem):
        return results[0], context
    else:
        return results, context

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
    engine=TfEngine(),
    silent=False
):  
    deduced, context = _deduce_elements([elem] + monitor.elems, feed_dict, structure, batch_size, False, silent, None, engine)

    to_optimize = deduced[0]

    to_monitor = OrderedDict()
    for m, val in zip(monitor.elems, deduced[1:]):
        name = m.get_name() if hasattr(m, "get_name") else str(m)
        to_monitor[name] = val

    opt_output = context.engine.optimize(to_optimize, optimizer, learning_rate)

    monitoring_values = []

    logging.info("Optimizing provided value for {} epochs using {} optimizer".format(epochs, optimizer))
    for e in xrange(epochs):
        returns = _run(context.engine, [to_optimize, opt_output], feed_dict, batch_size)

        
        if e % monitor.freq == 0:
            monitor_returns = _run(
                context.engine,
                [to_optimize] + to_monitor.values(), 
                monitor.feed_dict if not monitor.feed_dict is None else feed_dict,
                batch_size
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

    return returns[0], np.asarray(monitoring_values), context
