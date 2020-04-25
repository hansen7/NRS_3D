"""
Created on 2019.03.05

@author: caoyh
"""

'''
dd = 2048 for resnet50
nMul = 2 is good for fine-grained datasets, e.g. CUB-200 and Aircrafts
nPer you may try it for 16, 32, 64, 128, I use 64 for resnet50 on CUB200
dH/dW : when dH/dW=3, there is one group conv layer and this is good for fewer parameters
        when dH/dW=5, there is two group conv layers, accuracies sometimes become better but the parameters and FLOPs also get larger
'''

#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import torch, torchvision, sys
import torch.nn as nn, numpy as np
from copy import deepcopy

sys.path.append('../')

__all__ = ['resnet50_nfl', 'resnet50_baseline']


def perm_mask(dd, dH, dW, nMul):
	# generating a mask for permutation into dH x dW x (dd*nMul) tensor
	m = np.random.permutation(dd)
	# m = np.arange(dd)
	for i in range(1, dH * dW * nMul):
		m = np.concatenate((m, np.random.permutation(dd)))
	return m


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


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.bn3(out)
		if self.downsample is not None:
			identity = self.downsample(x)
		out += identity
		out = self.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, cfg, block, layers):
		super(ResNet, self).__init__()
		self.dd = cfg.MODEL.DD
		self.dH = cfg.MODEL.DH
		self.dW = cfg.MODEL.DW
		self.nMul = cfg.MODEL.N_MUL
		nFC = cfg.MODEL.FC.N_FC
		bFC = cfg.MODEL.FC.B_FC
		nClass = cfg.DATASETS.CLASS
		nPerGroup = cfg.MODEL.N_PER_GROUP

		mask = perm_mask(self.dd, self.dH, self.dW, self.nMul)
		self.register_buffer('mask', torch.from_numpy(mask))

		# resnet Stage 1
		self.inplanes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)

		# resnet maxpool
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		# resnet layer1
		self.layer1 = self._make_layer(block, 64, layers[0])

		# resnet layer2
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

		# resnet layer3
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

		# resnet layer4
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		# avgpool
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		self.nfl1 = nn.Sequential(
			nn.Conv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=3,
					  padding=0, groups=self.dd * self.nMul // nPerGroup),
			nn.BatchNorm2d(self.dd * self.nMul),
			nn.ReLU()
		)

		if self.dH > 3:
			self.nfl2 = nn.Sequential(
				nn.Conv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=3,
						  padding=0, groups=self.dd * self.nMul // nPerGroup),
				nn.BatchNorm2d(self.dd * self.nMul),
				nn.ReLU()
			)
		else:
			self.nfl2 = nn.Sequential()

		# self.nfl3 = nn.Sequential(
		#    nn.Conv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=3,
		#                    padding=0, groups=self.dd * self.nMul // nPerGroup),
		# nn.Conv2d(dd*nMul,dd*nMul,dH,1,0,1,dd*nMul//20),
		#    nn.BatchNorm2d(self.dd * self.nMul),
		#    nn.ReLU()
		# )

		if bFC:
			self.dense = nn.Sequential(
				nn.Linear(self.dd * self.nMul, nFC),
				nn.BatchNorm1d(nFC),
				nn.ReLU(),
				# nn.Linear(nFC, nFC),
				# nn.BatchNorm1d(nFC),
				# nn.ReLU(),
				nn.Linear(nFC, nClass)
			)
		else:
			self.dense = nn.Sequential(
				nn.Linear(self.dd * self.nMul, nClass)
			)
		'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion)
			)
		layers = [block(self.inplanes, planes, stride, downsample)]
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)

		# the original code is:
		# x = self.fc(x)
		# return x

		'''=== NFL Module ==='''
		x = torch.stack([xi[self.mask] for xi in torch.unbind(x, dim=0)], dim=0)
		x = x.view(x.size(0), self.dd * self.nMul, self.dH, self.dW)

		# x = x.view(x.size(0), self.dH, self.dW, self.dd*self.nMul)
		# x = x.permute(0,3,1,2)
		x = self.nfl1(x)
		x = self.nfl2(x)
		# x = self.nfl3(x)

		x = x.view(x.size(0), -1)
		x = self.dense(x)

		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


def resnet50_baseline(cfg, pretrained=False):
	model = torchvision.models.resnet50(num_classes=cfg.DATASETS.CLASS)
	model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
	if pretrained:
		pretrained_model = torchvision.models.resnet50(pretrained=True)
		model = copy_parameters(model, pretrained_model)
	return model


def resnet50_nfl(cfg, pretrained=False):
	model = ResNet(cfg, Bottleneck, [3, 4, 6, 3])
	if pretrained:
		pretrained_model = torchvision.models.resnet50(pretrained=True)
		model = copy_parameters(model, pretrained_model)
	return model
