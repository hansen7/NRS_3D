#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import torch
from thop import profile
from ref.resnet_nfl import resnet50_nfl
from models.pointnetnfl_cls import get_model
from models.pointnet2nfl_cls_ssg import get_model
# from models.pointnet2_cls_msg import get_model
from data_utils.Dict2Object import Dict2Object
from dgcnn import DGCNN, DGCNN_NFL
from train_dgcnn_cls import parse_args
# from Pointnet_Pointnet2_pytorch.models.pointnet_cls import get_model
# from Pointnet_Pointnet2_pytorch.models.pointnetnfl_new_cls import get_model


if __name__ == "__main__":

	# model = resnet50()
	# model = resnet50_nfl(cfg=Dict2Object('../pointnet_nfl.yaml'))
	# input = torch.randn(1, 3, 224, 224)

	model = get_model(num_class=40,
					  nfl_cfg=Dict2Object('../nfl_config/pointnetnfl_cls.yaml'))
	# input = torch.randn(1, 3, 1024)

	args = parse_args()

	args.nfl_cfg = '../nfl_config/dgcnnnfl_cls.yaml'
	model = DGCNN(args)
	# model = DGCNN_NFL(args)
	input = torch.randn(1, 3, 1024)

	macs, params = profile(model, inputs=(input, ))

	# print(macs)
	print('\n======================')
	print('FLOPs: %.1f G' % (macs/1e9))
	print('Num of Params %.1f M' % (params/1e6))
