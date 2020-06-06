#!/bin/bash

mkdir data
cd data
wget "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
unzip modelnet40_normal_resampled.zip
rm modelnet40_normal_resampled.zip