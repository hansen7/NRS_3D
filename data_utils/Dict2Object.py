#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
import yaml


class Dict2Object(object):
	def __init__(self, Path2Dict=None):
		if (type(Path2Dict) is str) and (Path2Dict.split('.')[-1] == 'yaml' or 'yml'):
			Path2Dict = yaml.load(open(Path2Dict), Loader=yaml.FullLoader)

		if type(Path2Dict) is dict:
			for key, value in Path2Dict.items():
				if type(value) is dict:
					setattr(self, key, Dict2Object(value))
				else:
					setattr(self, key, value)
		else:
			raise ValueError('Dict2Object only support input format as dictionary and .yaml/.yml')
