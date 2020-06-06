#!/bin/bash

python train_cls.py \
	--gpu None \
	--log_dir timer \
	--model pointnet2nrs_cls_ssg \
	--nrs_cfg pointnet_cls \
	--inference_timer 

