#!/bin/bash

python train_cls.py \
	--log_dir pointnet2nrs_cls_ssg \
	--model pointnet2nrs_cls_ssg \
	--nrs_cfg pointnet_cls \
	--normal \
	--gpu 7;