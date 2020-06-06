#!/bin/bash

python train_dgcnn_cls.py \
	--log_dir dgcnn_nfl_dh3_nmul1_nper128_1 \
	--nfl_cfg dgcnn_nfl_dh3_nmul1_nper128 \
	--gpu 7 \
	--model dgcnn_nfl \
	--data_aug \
	--use_sgd;

python train_dgcnn_cls.py \
	--log_dir dgcnn_nfl_dh3_nmul1_nper64_1 \
	--nfl_cfg dgcnn_nfl_dh3_nmul1_nper64 \
	--gpu 7 \
	--model dgcnn_nfl \
	--data_aug \
	--use_sgd;

python train_dgcnn_cls.py \
	--log_dir dgcnn_nfl_dh3_nmul1_nper32_1 \
	--nfl_cfg dgcnn_nfl_dh3_nmul1_nper32 \
	--gpu 7 \
	--model dgcnn_nfl \
	--data_aug \
	--use_sgd;
