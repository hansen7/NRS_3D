#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang, modified by Hanchen Wang, 2020
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""

from __future__ import print_function
import sys, os, pdb, torch, argparse, numpy as np, torch.nn as nn, torch.optim as optim, sklearn.metrics as metrics
sys.path.append('../')
sys.path.append('../data_utils')
from pc_utils import random_point_dropout, random_scale_point_cloud, random_shift_point_cloud
from torch.optim.lr_scheduler import CosineAnnealingLR
from ModelNetDataLoader import ModelNetDataLoader
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
from model import DGCNN, DGCNN_NFL
from tqdm import tqdm

DATA_PATH = 'data/modelnet40_normal_resampled/'

def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.log_dir):
		os.makedirs('checkpoints/' + args.log_dir)
	if not os.path.exists('checkpoints/' + args.log_dir + '/' + 'models'):
		os.makedirs('checkpoints/' + args.log_dir + '/' + 'models')
	os.system('cp main.py checkpoints' + '/' + args.log_dir + '/' + 'main.py.backup')
	os.system('cp model.py checkpoints' + '/' + args.log_dir + '/' + 'model.py.backup')
	os.system('cp util.py checkpoints' + '/' + args.log_dir + '/' + 'util.py.backup')
	os.system('cp data.py checkpoints' + '/' + args.log_dir + '/' + 'data.py.backup')


def train(args, io):
	# Load Data (includes normals)
	TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_points, split='train')
	TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_points, split='test')
	train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
	test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
	device = torch.device("cuda" if args.cuda else "cpu")

	# Load Models, Allow Multiple GPUs
	if args.model == 'dgcnn':
		model = DGCNN(args).to(device)
	elif args.model == 'dgcnn_nfl':
		model = DGCNN_NFL(args).to(device)
	else:
		raise Exception("Specified Model is Not Implemented")
	model = nn.DataParallel(model)

	if args.use_sgd:
		print("Use SGD")
		opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
	else:
		print("Use Adam")
		opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

	scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
	criterion, best_test_acc = cal_loss, 0

	for epoch in range(1, args.epochs+1):

		'''=== Train ==='''
		scheduler.step()
		train_loss, count = 0., 0.
		train_pred, train_true = [], []
		model.train()

		for batch_id, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
			pdb.set_trace()
			points, label = data
			# data, label = data.to(device), label.to(device).squeeze()
			points = data.permute(0, 2, 1)

			if args.data_aug:
				points = points.data.numpy()
				points = random_point_dropout(points)
				points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
				points[:, :, 0:3] = random_shift_point_cloud(points[:, :, 0:3])

			points, target = torch.Tensor(points).transpose(2, 1).cuda(), target[:, 0].cuda()
			batch_size = points.size()[0]

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
		outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (
			epoch,
			train_loss * 1.0 / count,
			metrics.accuracy_score(train_true, train_pred),
			metrics.balanced_accuracy_score(train_true, train_pred))

		io.cprint(outstr)

		'''=== Test ==='''
		test_loss, count = 0., 0.
		model.eval()
		test_pred, test_pred = [], []

		for data, label in test_loader:
			data, label = data.to(device), label.to(device).squeeze()
			data = data.permute(0, 2, 1)
			batch_size = data.size()[0]
			logits = model(data)
			loss = criterion(logits, label)
			preds = logits.max(dim=1)[1]
			count += batch_size
			test_loss += loss.item() * batch_size
			test_true.append(label.cpu().numpy())
			test_pred.append(preds.detach().cpu().numpy())
		test_true = np.concatenate(test_true)
		test_pred = np.concatenate(test_pred)
		test_acc = metrics.accuracy_score(test_true, test_pred)
		avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
		outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
																			  test_loss * 1.0 / count,
																			  test_acc,
																			  avg_per_class_acc)
		io.cprint(outstr)
		if test_acc > best_test_acc:
			best_test_acc = test_acc
			torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.log_dir)


def test(args, io):

	TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_points,
									  split='test', normal_channel=True)
	test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

	device = torch.device("cuda" if args.cuda else "cpu")

	# Try to load models
	model = DGCNN(args).to(device)
	model = nn.DataParallel(model)
	model.load_state_dict(torch.load(args.model_path))
	model = model.eval()
	test_true, test_pred = [], []

	for data, label in test_loader:
		data, label = data.to(device), label.to(device).squeeze()
		data = data.permute(0, 2, 1)
		logits = model(data)
		preds = logits.max(dim=1)[1]
		test_true.append(label.cpu().numpy())
		test_pred.append(preds.detach().cpu().numpy())
	test_true = np.concatenate(test_true)
	test_pred = np.concatenate(test_pred)
	test_acc = metrics.accuracy_score(test_true, test_pred)
	avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
	outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
	io.cprint(outstr)


if __name__ == "__main__":

	''' Parse Args for Training'''
	parser = argparse.ArgumentParser(description='Point Cloud Recognition')

	parser.add_argument('--model', type=str, default='dgcnn',
						choices=['dgcnn', 'dgcnn_nfl'],
						help='Model to use, [dgcnn, dgcnn_nfl]')
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

	''' Create Log Folders and Backup Scripts'''
	_init_(args)
	io = IOStream(r'./checkpoints/' + args.log_dir + '/run.log')
	io.cprint(str(args))
	torch.manual_seed(args.seed)

	if torch.cuda.is_available():
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
		io.cprint("Let's use" + str(torch.cuda.device_count()) + "GPUs!")
		torch.cuda.manual_seed(args.seed)
	else:
		io.cprint('Using CPU')

	''' Train the Model'''
	train(args, io)
