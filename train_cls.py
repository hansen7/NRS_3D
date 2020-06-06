#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import os, sys, pdb, time, torch, shutil, importlib, argparse, numpy as np
sys.path.append('utils')
sys.path.append('models')
from PC_Augmentation import random_point_dropout, random_scale_point_cloud, random_shift_point_cloud
from ModelNetDataLoader import ModelNetDataLoader
from torch.utils.tensorboard import SummaryWriter
from Inference_Timer import Inference_Timer
from Dict2Object import Dict2Object
from TrainLogger import TrainLogger
from tqdm import tqdm


def parse_args():
	parser = argparse.ArgumentParser('Point Cloud Classification')

	parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
	parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
	parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
	parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
	parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
	parser.add_argument('--inference_timer', action='store_true', default=False, help='use inference timer')
	parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
	parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
	parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
	parser.add_argument('--nrs_cfg', type=str, default='pointnet_cls', help='nrs configs [default: pointnet_cls]')
	parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normals [default: False]')
	parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate [default: 0.001]')

	return parser.parse_args()


def main(args):

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	''' === Inference Time Calculation === '''
	if args.inference_timer:
		MyTimer = Inference_Timer(args)
		args = MyTimer.update_args()  # Set the batch size as 1, and epoch as 3
	
	''' === Set up Loggers and Load Data === '''
	MyLogger = TrainLogger(args, name=args.model.upper(), subfold='cls')
	MyLogger.logger.info('Load dataset ...')
	DATA_PATH = 'data/modelnet40_normal_resampled/'
	TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)
	TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
	trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
	testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

	''' === Model Loading and Files Backup === '''
	MODEL = importlib.import_module(args.model)
	shutil.copy(args.nrs_cfg, MyLogger.log_dir)
	shutil.copy(os.path.abspath(__file__), MyLogger.log_dir)
	shutil.copy('./models/%s.py' % args.model, MyLogger.log_dir)
	writer = SummaryWriter(os.path.join(MyLogger.experiment_dir, 'runs'))

	# allow multiple GPU running
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	classifier = MODEL.get_model(num_class=40, normal_channel=args.normal, nrs_cfg=Dict2Object(args.nrs_cfg)).to(device)
	criterion = MODEL.get_loss().to(device)
	classifier = torch.nn.DataParallel(classifier)
	print("="*33, "\n", "Let's use", torch.cuda.device_count(), "GPUs, Indices are: %s!" % args.gpu, "\n", "="*33)
	
	try:
		checkpoint = torch.load(MyLogger.savepath)
		classifier.load_state_dict(checkpoint['model_state_dict'])
		MyLogger.update_from_checkpoints(checkpoint)
	except:
		MyLogger.logger.info('No pre-trained model, start training from scratch...')

	''' === Optimiser and Scheduler === '''
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
		''' === Training === '''
		scheduler.step()
		classifier.train()
		MyLogger.cls_epoch_init()
		# writer.add_scalar('learning rate', scheduler.get_lr()[-1], global_step)

		for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
			# Data Augmentation
			points = random_point_dropout(points.data.numpy())
			points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
			points[:, :, 0:3] = random_shift_point_cloud(points[:, :, 0:3])
		
			if args.gpu == 'None':
				points, target = torch.Tensor(points).transpose(2, 1), target[:, 0]
			else:
				points, target = torch.Tensor(points).transpose(2, 1).cuda(), target[:, 0].cuda()
			
			# FP and BP
			optimizer.zero_grad()
			if args.inference_timer:
				pred, trans_feat = MyTimer.single_step(classifier, points)
			else:
				pred, trans_feat = classifier(points)

			loss = criterion(pred, target.long(), trans_feat)
			loss.backward()
			optimizer.step()
			MyLogger.cls_step_update(pred.data.max(1)[1].cpu().numpy(), 
									 target.long().cpu().numpy(), 
									 loss.cpu().detach().numpy())
		MyLogger.cls_epoch_summary(writer=writer, training=True)
		MyTimer.update_single_epoch(MyLogger.logger)

		'''Validating'''
		with torch.no_grad():
			classifier.eval()
			MyLogger.cls_epoch_init(training=False)
			for points, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
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
	MyLogger.cls_train_summary()


if __name__ == '__main__':
	''' Parse Args for Training'''
	args = parse_args()
	args.nrs_cfg = os.path.join('nrs_cfg', args.nrs_cfg + '.yaml')

	''' Train the Model'''
	main(args)
