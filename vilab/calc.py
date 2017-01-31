
from engine import Engine
from api import *

import logging
import copy
import numbers
import types

def deduce(model, output, dependencies, feed_dict, structure, batch_size, engine_inputs):
	visited = {}
	calculated = {}

	def deduce_recurrent(elem, requested_shape = None):
		dst_elem = visited.get(elem)
		if not dst_elem is None:
			logging.debug("Element {} is already visited. Ignoring ...".format(elem))
			return dst_elem

		if isinstance(elem, Density):
			logging.debug("Deducer({}): deducing for density".format(elem))
			dst_elem = copy.copy(elem)
			dst_elem.set_args(
				[deduce_recurrent(a, requested_shape) for a in elem.get_args()]
			)
			assert not requested_shape is None, "Shape information is not provided to sample {}".format(dst_elem)

			logging.info("Sampling {} with shape {}x{}".format(elem, batch_size, requested_shape))
			dst_elem = Engine.sample(dst_elem, (batch_size, requested_shape))
		elif isinstance(elem, Variable):
			logging.debug("Deducer({}): deducing for variable".format(elem))
			assert elem == output or elem in dependencies, "Can't find variable {} as dependent in model {}".format(elem, model)
			
			if elem in feed_dict:
				logging.debug("Deducer({}): Found variable in inputs".format(elem))
			
				data = feed_dict[elem]
				provided_input = Engine.provide_input(elem.get_name(), data.shape)
				assert not provided_input in engine_inputs, "Visiting input for {} again".format(elem.get_name())
				engine_inputs[provided_input] = data
				dst_elem = provided_input
			else:
				rec = model.get_variable_record(elem)
				logging.debug("Deducer({}): got variable record {}".format(elem, rec))
				
				elem_structure = structure.get(elem)
				assert not elem_structure is None, "Need to provide structure information for {}".format(elem)
				
				for d in rec.dependencies:
					logging.debug("Deducer({}): deducing dependency {}".format(elem, d))
					deduce_recurrent(d)
				logging.debug("Deducer({}): deducing density {}".format(elem, rec.density))
				
				logging.info("Deducing variable {}".format(elem.get_name()))

				dst_elem = deduce_recurrent(rec.density, elem_structure)

		elif isinstance(elem, FunctionCallee):
			logging.debug("Deducer({}): Calling function".format(elem))
			deduced_args = [deduce_recurrent(arg) for arg in elem.get_args()]

			if isinstance(elem.get_fun(), Function):
				elem_structure = structure.get(elem.get_fun(), requested_shape)
				assert not elem_structure is None, "Need to provide structure information for {}".format(elem)
								
				logging.info("Calling function {} with structure {}, arguments: {}".format(
					elem.get_name(), elem_structure, ",".join([str(a.get_name()) for a in elem.get_args()])
				))
				# requested_shape = elem.get_size()[0]
				dst_elem = Engine.function(
					*deduced_args, 
					size = elem_structure, 
					name = "{}/{}".format(model.get_name(), elem.get_name()),
					act = deduce_recurrent(elem.get_act())
				)
			elif isinstance(elem, BasicFunction):
				dst_elem = Engine.basic_function(elem, *deduced_args)
			else:
				raise Exception("Unknown type of function: {}".format(elem.get_fun()))
		elif isinstance(elem, numbers.Integral):
			dst_elem = elem
		elif isinstance(elem, numbers.Real):
			dst_elem = elem
		elif isinstance(elem, Metrics):
			dst_elem = Engine.calculate_metrics(elem)
		else:
			raise Exception("Deducer met unexpected type: {}".format(elem))
		visited[elem] = dst_elem
		return dst_elem


	return deduce_recurrent(output)


def sample(model_slice, feed_dict={}, structure={}, batch_size=1):
	engine_inputs = {}
	
	returns = []
	logging.debug("Sampling {}".format(model_slice))
	
	model, output, dependencies = model_slice.get_slice_info()
	
	result = deduce(model, output, dependencies, feed_dict, structure, batch_size, engine_inputs)
	logging.debug("Collected {}".format(result))
	returns.append(result)
	return Engine.run(
		returns, 
		engine_inputs
	)


