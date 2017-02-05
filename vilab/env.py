
import os
from os.path import join as pj

def make_dir(*args):
	if not os.path.exists(pj(*args)):
		os.makedirs(pj(*args))
	return pj(*args)


class Env(object):
	ROOT_PATH = pj(os.environ["HOME"], "vilab")

	def __init__(self, project="default", root = ROOT_PATH, clear_pics=False):
		root = pj(root, project)
		
		self._res_dir = make_dir(root, "result")
		self._dataset_dir = make_dir(root, "dataset")
		self._run_dir = make_dir(root, "run")
		if clear_pics:
			self.clear_pics(self.run())
			self.clear_pics(self.result())
			self.clear_pics(self.dataset())

	def result(self, *args):
		if len(args)>0:
			return pj(self._res_dir, *args)
		return self._res_dir

	def dataset(self, *args):
		if len(args)>0:
			return pj(self._dataset_dir, *args)
		return self._dataset_dir

	def run(self, *args):
		if len(args)>0:
			return pj(self._run_dir, *args)
		return self._run_dir

	def clear_pics(self, d):
		[ os.remove(pj(d, f)) for f in os.listdir(d) if f[-4:] == ".png" ]


