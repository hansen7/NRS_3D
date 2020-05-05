#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import torch.utils.data, sys, os, pdb, torch.nn as nn, torch.nn.functional as F
sys.path.append('../')
from pointnet import PointNetEncoder, feature_transform_reguliarzer
from data_utils.Dict2Object import Dict2Object
from torchsummary import summary
from NFL import NFL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class get_model(nn.Module):
	def __init__(self, nfl_cfg, k=40, normal_channel=False):
		super(get_model, self).__init__()
		if normal_channel:
			channel = 6
		else:
			channel = 3
		self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
		self.nfl = NFL(nfl_cfg)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, k)

		# set dropout ratio as 0.4
		self.dropout = nn.Dropout(p=0.4)
		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)
		self.relu = nn.ReLU()

	def forward(self, x):
		x, trans, trans_feat = self.feat(x)
		x = self.nfl(x)
		# pdb.set_trace()
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.dropout(self.fc2(x))))
		x = self.fc3(x)
		x = F.log_softmax(x, dim=1)

		return x, trans_feat

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


class get_loss(torch.nn.Module):
	def __init__(self, mat_diff_loss_scale=0.001):
		super(get_loss, self).__init__()
		self.mat_diff_loss_scale = mat_diff_loss_scale

	def forward(self, pred, target, trans_feat):
		loss = F.nll_loss(pred, target)
		mat_diff_loss = feature_transform_reguliarzer(trans_feat)

		total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
		return total_loss


if __name__ == '__main__':
	# 24, 1024, 3

	cfg = Dict2Object(Path2Dict='../nfl_config/pointnetnfl_cls.yaml')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = get_model(nfl_cfg=cfg).to(device)
	# https://github.com/sksq96/pytorch-summary/issues/33
	summary(model, (3, 1024))
