#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import sys, os, pdb, torch, argparse
from thop import profile
sys.path.append('../')
sys.path.append('../models')
from models.pointnetnfl_cls import get_model
from models.pointnet2nfl_cls_ssg import get_model
# from models.pointnet2_cls_ssg import get_model
from data_utils.Dict2Object import Dict2Object
from models.dgcnn import DGCNN, DGCNN_NFL


if __name__ == "__main__":

	''' === PointNet and PointNet++ === '''
	model = get_model(k=40, nfl_cfg=Dict2Object(
		'../nfl_config/pointnet_nfl_dh3_nmul1_nper32.yaml')).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

	''' === DGCNN === '''
	def parse_args():
		parser = argparse.ArgumentParser(description='Point Cloud Recognition')
		parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
		parser.add_argument('--batch_size', type=int, default=1, help='Training Batch Size')
		parser.add_argument('--k', type=int, default=20, help='Num of nearest neighbors to use')
		parser.add_argument('--emb_dims', type=int, default=1024, help='Dimension of Embeddings')
		parser.add_argument('--num_point', type=int, default=1024, help='num of points of each object')
		parser.add_argument('--nfl_cfg', type=str, default='../nfl_config/dgcnn_nfl_dh3_nmul1_nper128.yaml', 
							help='config for NFL modules')
		return parser.parse_args()
	
	args = parse_args()
	model = DGCNN(args).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
	model = DGCNN_NFL(args).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
	
	''' === Set up the Shape of Input === '''
	input = torch.randn(1, 3, 1024).cuda()
	# input = torch.randn(1, 6, 1024).cuda()
	macs, params = profile(model, inputs=(input, ))

	print('\n======================')
	print('FLOPs: %.1f G' % (macs/1e9))
	print('Num of Params %.1f M' % (params/1e6))
