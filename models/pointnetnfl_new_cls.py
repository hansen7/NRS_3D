#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import torch.utils.data, os, sys, pdb, torch.nn as nn, torch.nn.functional as F
sys.path.append(['./', '../'])
from pointnet import PointNetEncoder, feature_transform_reguliarzer
from NFL import NFL
from utils.Dict2Object import Dict2Object

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class get_model(nn.Module):
	def __init__(self, k=40, normal_channel=False, cfg=Dict2Object(os.path.join(ROOT_DIR, 'pointnetnfl_cls.yaml'))):
		super(get_model, self).__init__()
		if normal_channel:
			channel = 6
		else:
			channel = 3
		self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
		self.nfl = NFL(cfg)
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
		# x = F.relu(self.bn1(self.fc1(x)))
		# x = F.relu(self.bn2(self.dropout(self.fc2(x))))
		# x = self.fc3(x)
		# x = F.log_softmax(x, dim=1)

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


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


if __name__ == '__main__':
	# 24, 1024, 3

	cfg = Dict2Object(Path2Dict='../pointnetnfl_cls.yaml')
	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# model = get_model(cfg=cfg).to(device)
	# # https://github.com/sksq96/pytorch-summary/issues/33
	# summary(model, (3, 1024))
