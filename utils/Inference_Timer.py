#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import os, torch, time, numpy as np

class Inference_Timer:
	def __init__(self, args):
		self.args = args
		self.est_total = []
		self.use_cpu = True if (self.args.gpu == 'None') else False
		if self.use_cpu:
			os.environ['OMP_NUM_THREADS'] = "10"
			os.environ['MKL_NUM_THREADS'] = "10"
			print('Now we calculate the inference time on a single CPU')
		else:
			print('Now we calculate the inference time on a single GPU')
		self.args.batch_size, self.args.epoch = 2, 3
		#  we cannot set batch_size as 1, since it BatchNorm requires the more than one sample to compute std
		#  red: https://github.com/pytorch/pytorch/issues/7716

	def update_args(self):
		return self.args

	def single_step(self, model, data):
		if not self.use_cpu:
			torch.cuda.synchronize()
		start = time.time()
		output = model(data)
		if not self.use_cpu:
			torch.cuda.synchronize()
		end = time.time()
		self.est_total.append(end - start)
		return output

	def update_single_epoch(self, logger):
		logger.info("Inference Time Per Examples: {}".format(np.mean(self.est_total)))
