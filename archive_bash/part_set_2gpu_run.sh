#!/bin/bash

screen -S pointnet_wnormal_vanilla -dm python train_partseg.py --log_dir pointnet_wnormal_vanilla --gpu 6 --model pointnet_part_seg --normal &
screen -S pointnet_nfl_dh3_nmul2_nper32 -dm python train_partseg.py --log_dir pointnet_nfl_dh3_nmul2_nper32 --gpu 7 --model pointnetnfl_part_seg --normal &
wait
screen -S pointnet2_msg_wnormal_vanilla -dm python train_partseg.py --log_dir pointnet2_msg_wnormal_vanilla --gpu 6,7 --normal
