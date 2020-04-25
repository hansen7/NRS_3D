#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

from resnet_nfl import perm_mask
import torch, torchvision
import torch.nn as nn


class VGG(nn.Module):
	def __init__(self, cfg, init_weights=True):
		super(VGG, self).__init__()
		self.dd = cfg.MODEL.DD
		self.dH = cfg.MODEL.DH
		self.dW = cfg.MODEL.DW
		self.nMul = cfg.MODEL.N_MUL
		nFC = cfg.MODEL.FC.N_FC
		bFC = cfg.MODEL.FC.B_FC
		nClass = cfg.DATASETS.CLASS
		nPerGroup = cfg.MODEL.N_PER_GROUP

		mask = perm_mask(self.dd, self.dW, self.dH, self.nMul)

		self.register_buffer('mask', torch.from_numpy(mask))

		self.features = torchvision.models.vgg16().features

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		self.nfl1 = nn.Sequential(
			nn.Conv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=3,
					  padding=0, groups=self.dd * self.nMul // nPerGroup),
			# nn.Conv2d(dd*nMul,dd*nMul,dH,1,0,1,dd*nMul//20),
			nn.BatchNorm2d(self.dd * self.nMul),
			nn.ReLU()
		)

		if self.dH > 3:
			self.nfl2 = nn.Sequential(
				nn.Conv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=3,
						  padding=0, groups=self.dd * self.nMul // nPerGroup),
				# nn.Conv2d(dd*nMul,dd*nMul,dH,1,0,1,dd*nMul//20),
				nn.BatchNorm2d(self.dd * self.nMul),
				nn.ReLU()
			)
		else:
			self.nfl2 = nn.Sequential()
		# self.register_buffer('mask_buffer', torch.from_numpy(self.mask))

		# self.nfl3 = nn.Sequential(
		#    nn.Conv2d(in_channels=self.dd*self.nMul, out_channels=self.dd*self.nMul, kernel_size=3,
		#                    padding=0, groups=self.dd * self.nMul),
		#    # nn.Conv2d(dd*nMul,dd*nMul,dH,1,0,1,dd*nMul//20),
		#    nn.BatchNorm2d(self.dd * self.nMul),
		#    nn.ReLU()
		# )

		if bFC:
			self.dense = nn.Sequential(
				nn.Linear(self.dd * self.nMul, nFC),
				nn.BatchNorm1d(nFC),
				nn.ReLU(),
				nn.Linear(nFC, nFC),
				nn.BatchNorm1d(nFC),
				nn.ReLU(),
				nn.Linear(nFC, nClass)
			)
		else:
			self.dense = nn.Sequential(
				nn.Linear(self.dd * self.nMul, nClass)
				# nn.Linear(self.dd*self.nMul, nClass)
			)

	# if init_weights:
	#    self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)

		x = x.view(x.size(0), -1)

		# x = x.expand(x.size(0), x.size(1), )
		x = torch.stack([xi[self.mask] for xi in torch.unbind(x, dim=0)], dim=0)
		x = x.view(x.size(0), self.dd * self.nMul, self.dH, self.dW)
		# x = x.view(x.size(0), self.dH, self.dW, self.dd * self.nMul)
		# x = x.permute(0, 3, 1, 2)
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
