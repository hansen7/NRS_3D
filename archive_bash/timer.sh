#!/bin/bash

# Inference Time on Single CPU
python train_cls.py \
	--gpu None \
	--log_dir timer \
	--model pointnet2nrs_cls_ssg \
	--nrs_cfg pointnet_cls \
	--inference_timer

# Inference Time on Single GPU
python train_cls.py \
	--gpu 7 \
	--log_dir timer \
	--model pointnet2nrs_cls_ssg \
	--nrs_cfg pointnet_cls \
	--inference_timer

