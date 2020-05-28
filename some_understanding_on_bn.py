#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import torch, torch.nn as nn, pdb


class SomeNet(nn.Module):
	def __init__(self):
		super(SomeNet, self).__init__()
		self.bn1 = nn.BatchNorm1d(4)

	def forward(self, x):
		return self.bn1(x)


# model = nn.Sequential(
# 	nn.BatchNorm1d(4),
# )

rand_input = torch.randn(2, 4, 2)  # (N, C, L)
model = SomeNet()
model(rand_input)
# inputs = torch.Tensor(2,50,70)
