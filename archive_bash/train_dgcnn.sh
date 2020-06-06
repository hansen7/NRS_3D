#!/bin/bash

python train_dgcnn_cls.py \
	--log_dir dgcnn_nrs \
	--nfl_cfg dgcnn_cls \
	--model dgcnn_nrs \
	--data_aug \
	--use_sgd \
	--gpu 7;