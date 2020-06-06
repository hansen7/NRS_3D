#!/bin/bash

python train_cls.py \
	--log_dir pointnetnrs_cls \
	--model pointnetnrs_cls \
	--nrs_cfg pointnet_cls \
	--gpu 7;