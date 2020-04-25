#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision import datasets
import numbers, os, shutil, torch, torchvision, time, logging
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn, torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from resnet_nfl import resnet50_nfl

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])


def center_crop_with_flip(img, size, vertical_flip=False):
	crop_h, crop_w = size
	first_crop = F.center_crop(img, (crop_h, crop_w))
	if vertical_flip:
		img = F.vflip(img)
	else:
		img = F.hflip(img)
	second_crop = F.center_crop(img, (crop_h, crop_w))
	return first_crop, second_crop


class CenterCropWithFlip(object):
	def __init__(self, size, vertical_flip=False):
		self.size = size
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
			self.size = size
		self.vertical_flip = vertical_flip

	def __call__(self, img):
		return center_crop_with_flip(img, self.size, self.vertical_flip)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(self.size, self.vertical_flip)


def get_dataset():
	"""
    param cfg: dataset_name, cub200 or uci_dataset
    return: pytorch train_loader, test_loader
    """
	val_loader = None
	im_size = 224

	train_transforms = transforms.Compose([
		# transforms.Resize(size=im_size),
		transforms.Resize(size=(256, 256)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomCrop(size=im_size),
		transforms.ToTensor(),
		normalize
	])
	val_transforms = transforms.Compose([
		# transforms.Resize(size=im_size),
		transforms.Resize(size=(256, 256)),
		transforms.CenterCrop(size=im_size),
		transforms.ToTensor(),
		normalize
	])

	root = '/opt/caoyh/datasets/cub200/'

	MyTrainData = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transforms)
	MyValData = datasets.ImageFolder(os.path.join(root, 'val'), transform=val_transforms)

	train_loader = torch.utils.data.DataLoader(dataset=MyTrainData,
											   batch_size=64,
											   shuffle=True, num_workers=8, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(dataset=MyValData, batch_size=64,
											 num_workers=8, pin_memory=True)
	return train_loader, val_loader


def copy_parameters(model, pretrained):
	model_dict = model.state_dict()
	pretrained_dict = pretrained.state_dict()
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if
					   k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

	for k, v in pretrained_dict.items():
		print(k)

	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)
	return model


def resnet50_baseline(pretrained=False):
	model = torchvision.models.resnet50(num_classes=200)
	model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
	if pretrained:
		pretrained_model = torchvision.models.resnet50(pretrained=True)
		model = copy_parameters(model, pretrained_model)
	return model


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename[0])
	if is_best:
		shutil.copyfile(filename[0], filename[1])


def train(model):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
	scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model.to(device)

	start_epoch = 0
	best_acc = 0.
	num_epochs = 60

	cudnn.benchmark = True

	train_loader, val_loader = get_dataset()
	print("finish build model\n{}".format(model))

	loss_func = torch.nn.CrossEntropyLoss()
	loss_func.to(device)

	print("start training for {} epochs".format(num_epochs))

	is_best = False
	best_epoch = start_epoch
	eval_acc = 0.

	train_losses = []
	eval_losses = []
	train_accs = []
	eval_accs = []

	for epoch in range(start_epoch, num_epochs):
		train_loss = 0.
		train_acc = 0.
		train_total = 0.

		model.train()
		for batch_x, batch_y in train_loader:
			batch_x, batch_y = batch_x.to(device), batch_y.to(device)
			out = model(batch_x)

			loss = loss_func(out, batch_y)

			train_loss += loss.item()
			pred = torch.max(out, 1)[1]
			train_correct = (pred == batch_y).sum().item()
			train_acc += train_correct
			train_total += batch_y.size(0)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		train_loss = train_loss / train_total
		train_acc = 100 * float(train_acc) / float(train_total)

		scheduler.step()

		model.eval()
		eval_loss = 0.
		eval_acc = 0.
		eval_total = 0.
		for batch_x, batch_y in val_loader:
			with torch.no_grad():
				batch_x, batch_y = batch_x.to(device), batch_y.to(device)
				out = model(batch_x)
				loss = loss_func(out, batch_y)

				eval_loss += loss.item()
				pred = torch.max(out, 1)[1]
				num_correct = (pred == batch_y).sum().item()
				eval_acc += num_correct
				eval_total += batch_y.size(0)
		eval_loss = eval_loss / eval_total
		eval_acc = 100 * float(eval_acc) / float(eval_total)

		is_best = False
		if eval_acc > best_acc:
			is_best = True
			best_acc = eval_acc
			best_epoch = epoch + 1

		filename = []
		filename.append('checkpoint.pth.tar')
		filename.append('model_best.pth.tar')

		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'acc': eval_acc,
			# 'scheduler': scheduler.state_dict(),
			'optimizer': optimizer.state_dict(),
		}, is_best, filename)

		print_per_epoch = 1
		if (epoch + 1) % print_per_epoch == 0:
			print("epoch {}".format(epoch + 1))
			print("Train Loss: {:.6f}, Acc: {:.6f}%".format(train_loss, train_acc))
			print("Test Loss: {:.6f}, Acc: {:.6f}%".format(eval_loss, eval_acc))
			print('save model into {}'.format(filename[0]))
			train_losses.append(train_loss)
			eval_losses.append(eval_loss)
			train_accs.append(train_acc)
			eval_accs.append(eval_acc)

	print('Best at epoch %d, test_accuracy %f' % (best_epoch, best_acc))

	return model, eval_acc, best_acc


if __name__ == "__main__":
	model = resnet50_nfl(pretrained=True)
	train(model)
