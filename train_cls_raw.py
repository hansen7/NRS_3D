#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import os, sys, pdb, torch, shutil, importlib, argparse, h5py, numpy as np
sys.path.append('data_utils')
sys.path.append('models')
from pc_utils import random_point_dropout, random_scale_point_cloud, random_shift_point_cloud
from ModelNetDataLoader import ModelNetDataLoader
from torch.utils.tensorboard import SummaryWriter
from data_utils.TrainLogger import TrainLogger
from Dict2Object import Dict2Object
from tqdm import tqdm


def parse_args():
	parser = argparse.ArgumentParser('Point Cloud Classification')
	parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
	parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
	parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
	parser.add_argument('--decay_rate', type=float, default=0, help='decay rate [default: 0')
	parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
	parser.add_argument('--batch_size', type=int, default=32, help='batch size in training [default: 32]')
	parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
	parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
	parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normals [default: False]')
	parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
	parser.add_argument('--nfl_cfg', type=str, default='pointnetnfl_cls', help='NFL configs [default: pointnetnfl_cls]')
	return parser.parse_args()


def rotate_point_cloud(batch_data):
	""" Randomly rotate the point clouds to augment the dataset
		rotation is per shape based along up direction
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, rotated batch of point clouds
	"""
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	for k in range(batch_data.shape[0]):
		rotation_angle = np.random.uniform() * 2 * np.pi
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
		rotation_matrix = np.array([[cosval, 0, sinval],
		                            [0, 1, 0],
		                            [-sinval, 0, cosval]])
		shape_pc = batch_data[k, ...]
		rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
	return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
	""" Randomly jitter points. jittering is per point.
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, jittered batch of point clouds
	"""
	B, N, C = batch_data.shape
	assert(clip > 0)
	jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
	jittered_data += batch_data
	return jittered_data


def main(args):

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	'''Set up Loggers and Load Data'''
	MyLogger = TrainLogger(args, name=args.model.upper(), subfold='cls')
	MyLogger.logger.info('Load dataset ...')
	# DATA_PATH = 'data/modelnet40_normal_resampled/'
	# TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)
	# TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)

	TRAIN_DATASET = ModelNetH5Dataset(r'./data/modelnet40_ply_hdf5_2048/train_files.txt', batch_size=args.batch_size,
									  npoints=args.num_point, shuffle=True)
	TEST_DATASET = ModelNetH5Dataset(r'./data/modelnet40_ply_hdf5_2048/test_files.txt', batch_size=args.batch_size,
									  npoints=args.num_point, shuffle=False)

	# TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)

	# trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
	# testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

	'''Model Loading and Files Backup'''
	num_class = 40
	MODEL = importlib.import_module(args.model)
	shutil.copy(args.nfl_cfg, MyLogger.log_dir)
	shutil.copy(os.path.abspath(__file__), MyLogger.log_dir)
	shutil.copy('./models/%s.py' % args.model, MyLogger.log_dir)
	writer = SummaryWriter(os.path.join(MyLogger.experiment_dir, 'runs'))

	# allow multiple GPU running
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	classifier = MODEL.get_model(k=num_class, normal_channel=args.normal, nfl_cfg=Dict2Object(args.nfl_cfg)).to(device)
	criterion = MODEL.get_loss().to(device)
	classifier = torch.nn.DataParallel(classifier)
	print("="*33, "\n", "Let's use", torch.cuda.device_count(), "GPUs, Indices are: %s!" % args.gpu, "\n", "="*33)
	
	try:
		checkpoint = torch.load(MyLogger.savepath)
		classifier.load_state_dict(checkpoint['model_state_dict'])
		MyLogger.update_from_checkpoints(checkpoint)
	except:
		MyLogger.logger.info('No pre-trained model, start training from scratch...')

	'''Optimiser and Scheduler'''
	if args.optimizer == 'Adam':
		optimizer = torch.optim.Adam(
			classifier.parameters(),
			lr=args.learning_rate,
			betas=(0.9, 0.999),
			eps=1e-08,
			weight_decay=args.decay_rate
		)
	else:
		optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr*100, momentum=0.9)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

	for epoch in range(MyLogger.epoch, args.epoch + 1):
		'''Training'''
		scheduler.step()
		classifier.train()
		MyLogger.cls_epoch_init()
		# writer.add_scalar('learning rate', scheduler.get_lr()[-1], global_step)

		# for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
		# Data Augmentation
		while TRAIN_DATASET.has_next_batch():
			pdb.set_trace()
			points, target = TRAIN_DATASET.next_batch()
			rotated_data = rotate_point_cloud(points)
			jittered_data = jitter_point_cloud(rotated_data)

			# points = random_point_dropout(points.data.numpy())
			# points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
			# points[:, :, 0:3] = random_shift_point_cloud(points[:, :, 0:3])
			points, target = torch.Tensor(jittered_data).transpose(2, 1).cuda(), target[:, 0].cuda()

			# FP and BP
			optimizer.zero_grad()
			pred, trans_feat = classifier(points)
			loss = criterion(pred, target.long(), trans_feat)
			loss.backward()
			optimizer.step()
			MyLogger.cls_step_update(pred.data.max(1)[1].cpu().numpy(), 
									 target.long().cpu().numpy(), 
									 loss.cpu().detach().numpy())
		MyLogger.cls_epoch_summary(writer=writer, training=True)
		
		'''Validating'''
		with torch.no_grad():
			classifier.eval()
			MyLogger.cls_epoch_init(training=False)

			while TEST_DATASET.has_next_batch():
				points, target = TEST_DATASET.next_batch()
				points, target = points.transpose(2, 1).cuda(), target[:, 0].cuda()
				pred, trans_feat = classifier(points)
				loss = criterion(pred, target.long(), trans_feat)
				MyLogger.cls_step_update(pred.data.max(1)[1].cpu().numpy(),
										 target.long().cpu().numpy(),
										 loss.cpu().detach().numpy())
			
			MyLogger.cls_epoch_summary(writer=writer, training=False)
			if MyLogger.save_model:
				state = {
					'step': MyLogger.step,
					'epoch': MyLogger.best_instance_epoch,
					'instance_acc': MyLogger.best_instance_acc,
					'best_class_acc': MyLogger.best_class_acc,
					'best_class_epoch': MyLogger.best_class_epoch,
					'model_state_dict': classifier.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
				}
				torch.save(state, MyLogger.savepath)
		# pdb.set_trace()
	MyLogger.cls_train_summary()


class ModelNetH5Dataset(object):
	"""ModelNet dataset. Support ModelNet40, XYZ channels. Up to 2048 points.
    Faster IO than ModelNetDataset in the first epoch."""

	def __init__(self, list_filename, batch_size=32, npoints=1024, shuffle=True):
		self.list_filename = list_filename
		self.batch_size = batch_size
		self.npoints = npoints
		self.shuffle = shuffle
		self.h5_files = self.getDataFiles(self.list_filename)
		# self.reset()
		""" reset order of h5 files """
		self.file_idxs = np.arange(0, len(self.h5_files))
		if self.shuffle:
			np.random.shuffle(self.file_idxs)
		self.current_data = None
		self.current_label = None
		self.current_file_idx = 0
		self.batch_idx = 0

	@staticmethod
	def getDataFiles(list_filename):
		return [line.rstrip() for line in open(list_filename)]

	@staticmethod
	def load_h5(h5_filename):
		f = h5py.File(h5_filename, 'r')
		data = f['data'][:]
		label = f['label'][:]
		return data, label

	@staticmethod
	def shuffle_data(data, labels):
		idx = np.arange(len(labels))
		np.random.shuffle(idx)
		return data[idx, ...], labels[idx], idx

	# @staticmethod
	# def _augment_batch_data(batch_data):
	# 	rotated_data = data_utils.rotate_point_cloud(batch_data)
	# 	rotated_data = data_utils.rotate_perturbation_point_cloud(rotated_data)
	# 	jittered_data = data_utils.random_scale_point_cloud(rotated_data[:, :, 0:3])
	# 	jittered_data = data_utils.shift_point_cloud(jittered_data)
	# 	jittered_data = data_utils.jitter_point_cloud(jittered_data)
	# 	rotated_data[:, :, 0:3] = jittered_data
	# 	return data_utils.shuffle_points(rotated_data)

	def _get_data_filename(self):
		return self.h5_files[self.file_idxs[self.current_file_idx]]

	def _load_data_file(self, filename):
		self.current_data, self.current_label = self.load_h5(filename)
		self.current_label = np.squeeze(self.current_label)
		self.batch_idx = 0
		if self.shuffle:
			self.current_data, self.current_label, _ = self.shuffle_data(self.current_data, self.current_label)

	def _has_next_batch_in_file(self):
		return self.batch_idx * self.batch_size < self.current_data.shape[0]

	# def num_channel(self):
	# 	return 3

	def has_next_batch(self):
		if (self.current_data is None) or (not self._has_next_batch_in_file()):
			if self.current_file_idx >= len(self.h5_files):
				return False
			self._load_data_file(self._get_data_filename())
			self.batch_idx = 0
			self.current_file_idx += 1
		return self._has_next_batch_in_file()

	def next_batch(self):
		""" returned dimension may be smaller than self.batch_size """
		start_idx = self.batch_idx * self.batch_size
		end_idx = min((self.batch_idx + 1) * self.batch_size, self.current_data.shape[0])
		# bsize = end_idx - start_idx
		# batch_label = np.zeros(bsize, dtype=np.int32)
		data_batch = self.current_data[start_idx:end_idx, 0:self.npoints, :].copy()
		label_batch = self.current_label[start_idx:end_idx].copy()
		self.batch_idx += 1
		# if augment:
		# 	data_batch = self._augment_batch_data(data_batch)
		return data_batch, label_batch


if __name__ == '__main__':
	''' Parse Args for Training'''
	args = parse_args()
	args.nfl_cfg = os.path.join('nfl_config', args.nfl_cfg + '.yaml')

	''' Train the Model'''
	main(args)
