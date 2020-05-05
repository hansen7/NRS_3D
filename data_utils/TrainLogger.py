#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import logging, datetime, numpy as np, sklearn.metrics as metrics
from pathlib import Path


class TrainLogger:

	def __init__(self, args, name='Model', subfolder='classification', cls2name=None):
		self.step = 0
		self.epoch = 1
		self.args = args
		self.name = name
		self.logger_setup()
		self.cls_epoch_init()
		self.save_model = False
		self.cls2name = cls2name
		self.create_logdir(subfolder)
		self.best_instance_acc, self.best_class_acc = 0., 0.
		self.best_instance_epoch, self.best_class_epoch = 0, 0
		self.savepath = str(self.checkpoints_dir) + '/best_model.pth'

	def logger_setup(self):
		self.logger = logging.getLogger(self.name)
		self.logger.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		file_handler = logging.FileHandler('%s/%s.txt' % (self.args.log_dir, self.args.model))
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(formatter)
		self.logger.addHandler(file_handler)
		self.logger.info('PARAMETER ...')
		self.logger.info(self.args)

	def create_logdir(self, subfolder='classification'):
		timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
		experiment_dir = Path('./log/')
		experiment_dir.mkdir(exist_ok=True)
		experiment_dir = experiment_dir.joinpath(subfolder)
		experiment_dir.mkdir(exist_ok=True)

		if self.args.log_dir is None:
			self.experiment_dir = experiment_dir.joinpath(timestr)
		else:
			self.experiment_dir = experiment_dir.joinpath(self.args.log_dir)

		self.experiment_dir.mkdir(exist_ok=True)
		self.checkpoints_dir = experiment_dir.joinpath('checkpoints/')
		self.checkpoints_dir.mkdir(exist_ok=True)
		self.log_dir = experiment_dir.joinpath('logs/')
		self.log_dir.mkdir(exist_ok=True)

	# @property.setter
	def cls_epoch_init(self, training=True):
		self.loss, self.count, self.pred, self.gt = 0., 0., [], []
		if training:
			self.logger.info('\nEpoch %d/%d:' % (self.epoch, self.args.epoch))

	def cls_step_update(self, pred, gt, loss):
		self.step += 1
		self.gt.append(gt)
		self.pred.append(pred)
		batch_size = len(pred)
		self.count += batch_size
		self.loss += loss * batch_size

	def cls_epoch_update(self, training=True):
		self.save_model = False
		self.gt = np.concatenate(self.gt)
		self.pred = np.concatenate(self.pred)
		instance_acc = metrics.accuracy_score(self.gt, self.pred)
		class_acc = metrics.balanced_accuracy_score(self.gt, self.pred)

		if instance_acc > self.best_instance_acc and not training:
			self.best_instance_acc = instance_acc
			self.best_instance_epoch = self.epoch
			save_model = True
		if class_acc > self.best_class_acc and not training:
			self.best_class_acc = class_acc
			self.best_class_epoch = self.epoch

		if not training:
			self.epoch += 1
		return instance_acc, class_acc, save_model

	def cls_epoch_summary(self, writer, training=True):
		instance_acc, class_acc, save_model = self.cls_epoch_update(training)
		if training:
			writer.add_scalar('Train Class Accuracy', class_acc, self.step)
			writer.add_scalar('Train Instance Accuracy', instance_acc, self.step)
			self.logger.info('Train Instance Accuracy: %.3f, Class Accuracy: %.3f' % (instance_acc, class_acc))
		else:
			writer.add_scalar('Test Class Accuracy', class_acc, self.step)
			writer.add_scalar('Test Instance Accuracy', instance_acc, self.step)
			self.logger.info('Test Instance Accuracy: %.3f, Class Accuracy: %.3f, \
							  Best Instance Accuracy: %.3f at Epoch %d, \
							  Best Class Accuracy: %.3f at Epoch %d, ' % (
				instance_acc, class_acc, self.best_instance_acc, self.best_instance_epoch,
				self.best_class_acc, self.best_class_epoch))

		if save_model:
			self.logger.info('Saving the Model Params to %s' % self.savepath)
			self.save_model = True
		if not training:
			self.logger.info('\n')
		# return save_model

	def cls_train_summary(self):
		self.logger.info('End of Training...')
		self.logger.info('Best Instance Accuracy: %.3f at Epoch %d, Best Class Accuracy: %.3f at Epoch %d, ' % (
			self.best_instance_acc, self.best_instance_epoch, self.best_class_acc, self.best_class_epoch))

	def update_from_checkpoints(self, checkpoint):
		self.logger.info('Use Pre-Trained checkpoints')
		self.step = checkpoint['step']
		self.epoch = checkpoint['epoch']
		self.best_instance_epoch, self.best_instance_acc = checkpoint['epoch'], checkpoint['instance_acc']
		self.best_class_epoch, self.best_class_acc = checkpoint['best_class_epoch'], checkpoint['best_class_acc']
		self.logger.info('Best Class Acc {} at Epoch {}'.format(self.best_instance_acc, self.best_class_epoch))
		self.logger.info('Best Instance Acc {} at Epoch {}'.format(self.best_instance_acc, self.best_instance_epoch))
