#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
modified by Hanchen Wang, 2020
"""

import sys, os, pdb, torch, argparse, numpy as np, torch.nn as nn, torch.nn.functional as F, sklearn.metrics as metrics
sys.path.append('../')
sys.path.append('../data_utils')
from pc_utils import random_point_dropout, random_scale_point_cloud, random_shift_point_cloud
from ModelNetDataLoader import ModelNetDataLoader
from dgcnn import DGCNN, DGCNN_NFL
from tqdm import tqdm


def cal_loss(pred, gold, smoothing=True):
	""" Calculate cross entropy loss, apply label smoothing if needed. """
	gold = gold.contiguous().view(-1)
	
	if smoothing:
		eps, n_class = 0.2, pred.size(1)
		one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
		one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
		log_prb = F.log_softmax(pred, dim=1)
		loss = -(one_hot * log_prb).sum(dim=1).mean()  # mean
	else:
		loss = F.cross_entropy(pred, gold, reduction='mean')
	
	return loss


class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text+'\n')
		self.f.flush()

	def close(self):
		self.f.close()


def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.log_dir):
		os.makedirs('checkpoints/' + args.log_dir)
	if not os.path.exists('checkpoints/' + args.log_dir + '/' + 'models'):
		os.makedirs('checkpoints/' + args.log_dir + '/' + 'models')
	os.system('cp main.py checkpoints' + '/' + args.log_dir + '/' + 'main.py.backup')
	os.system('cp dgcnn.py checkpoints' + '/' + args.log_dir + '/' + 'model.py.backup')
	os.system('cp data.py checkpoints' + '/' + args.log_dir + '/' + 'data.py.backup')


def train(args):
	
	# Create Log Folders and Backup Scripts
	_init_(args)
	io = IOStream(r'./checkpoints/' + args.log_dir + '/run.log')
	io.cprint(str(args))
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	if torch.cuda.is_available():
		io.cprint("Let's use " + str(torch.cuda.device_count()) + " GPUs: " + args.gpu + "!")
		torch.cuda.manual_seed(args.seed)
	else:
		io.cprint('Using CPU')
	torch.manual_seed(args.seed)

	# Load Data (excludes normals)
	DATA_PATH = 'data/modelnet40_normal_resampled/'
	TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_points, split='train', normal_channel=False)
	TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_points, split='test', normal_channel=False)
	train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
	test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=6, shuffle=False, num_workers=4)
	# use smaller batch size in test_loader (no effect on training), just to make it executable on a single GTX 1080 (8GB Mem)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	# Load Models, Allow Multiple GPUs
	if args.model == 'dgcnn':
		model = DGCNN(args).to(device)
	elif args.model == 'dgcnn_nfl':
		model = DGCNN_NFL(args).to(device)
	else:
		raise Exception("Specified Model is Not Implemented")
	model = nn.DataParallel(model)

	if args.use_sgd:
		print("Use SGD Optimiser")
		opt = torch.optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
	else:
		print("Use Adam Optimiser")
		opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
	criterion, best_test_acc = cal_loss, 0

	for epoch in range(1, args.epochs+1):

		'''=== Train ==='''
		train_loss, count, train_pred, train_true = 0., 0., [], []
		scheduler.step()
		model.train()

		for _, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
			points, label = data  # points -> (batch_size, num_points, 3), 

			if args.data_aug:
				points = points.numpy()
				points = random_point_dropout(points)
				points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
				points[:, :, 0:3] = random_shift_point_cloud(points[:, :, 0:3])

			points, label = torch.Tensor(points).transpose(2, 1).cuda(), label[:, 0].type(torch.int64).cuda()  
			# (batch_size, 3, num_points)
			batch_size = points.size()[0]  # the last batch is smaller than args.batch_size

			opt.zero_grad()
			logits = model(points)
			loss = criterion(logits, label)
			loss.backward()
			opt.step()
			preds = logits.max(dim=1)[1]
			count += batch_size
			train_loss += loss.item() * batch_size
			train_true.append(label.cpu().numpy())
			train_pred.append(preds.detach().cpu().numpy())

		train_true = np.concatenate(train_true)
		train_pred = np.concatenate(train_pred)
		outstr = 'Train %d, Loss: %.6f, Train Instance Acc: %.6f, Train Avg Cls Acc: %.6f' % (
			epoch,
			train_loss * 1.0 / count,
			metrics.accuracy_score(train_true, train_pred),
			metrics.balanced_accuracy_score(train_true, train_pred))
		io.cprint(outstr)

		'''=== Test ==='''
		test_loss, count, test_pred, test_true = 0., 0., [], []
		model.eval()

		for _, data in tqdm(enumerate(test_loader, 0), total=len(test_loader), smoothing=0.9):
			points, label = data
			points, label = torch.Tensor(points).transpose(2, 1).cuda(), label[:, 0].type(torch.int64).cuda()
			batch_size = points.size()[0]
			logits = model(points)
			loss = criterion(logits, label)
			preds = logits.max(dim=1)[1]
			count += batch_size
			test_loss += loss.item() * batch_size
			test_true.append(label.cpu().numpy())
			test_pred.append(preds.detach().cpu().numpy())
		
		test_true, test_pred = np.concatenate(test_true), np.concatenate(test_pred)
		test_acc = metrics.accuracy_score(test_true, test_pred)

		if test_acc > best_test_acc:
			best_test_acc = test_acc
			torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.log_dir)
		
		outstr = 'Test %d, Loss: %.3f, Test Instance Acc: %.3f, Test Avg Cls Acc: %.3f, Best Instance Acc: %.3f' % (
			epoch,
			test_loss * 1.0 / count,
			metrics.accuracy_score(test_true, test_pred),
			metrics.balanced_accuracy_score(test_true, test_pred),
			best_test_acc)
		io.cprint(outstr)


if __name__ == "__main__":

	''' Parse Args for Training'''
	parser = argparse.ArgumentParser(description='Point Cloud Recognition')

	parser.add_argument('--model', type=str, default='dgcnn',
						choices=['dgcnn', 'dgcnn_nfl'],
						help='Model to use, [dgcnn, dgcnn_nfl]')
	parser.add_argument('--gpu', type=str, default='0', help='GPU')
	parser.add_argument('--log_dir', type=str, default='cls_vanilla', help='LOG')
	parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
	parser.add_argument('--batch_size', type=int, default=32, help='Training Batch Size')
	# parser.add_argument('--test_batch_size', type=int, default=16, help='Testing Batch Size')
	parser.add_argument('--epochs', type=int, default=250, help='number of training epochs')
	parser.add_argument('--k', type=int, default=20, help='Num of nearest neighbors to use')
	parser.add_argument('--emb_dims', type=int, default=1024, help='Dimension of Embeddings')
	parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	parser.add_argument('--num_points', type=int, default=1024, help='num of points of each object')
	parser.add_argument('--cfg', type=str, default='dgcnnnfl_cls.yaml', help='config for NFL modules')
	parser.add_argument('--model_path', type=str, default='', help='Pre-Trained model path, only used in test')
	parser.add_argument('--use_sgd', action='store_true', default=True, help='Use SGD Optimiser[default: True]')
	parser.add_argument('--data_aug', action='store_true', default=True, help='Data Augmentation[default: True]')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001, 0.1 if using sgd)')
	args = parser.parse_args()

	''' Train the Model'''
	train(args)
