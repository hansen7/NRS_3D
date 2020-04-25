"""
Data Preparation

Download 3D indoor parsing dataset (S3DIS) and save in data/Stanford3dDataset_v1.2/.

 - cd data_utils
 - `python collect_indoor3d_data.py` for data re-organization: convert txt into numpy.ndarray
 - `python gen_indoor3d_h5.py` to generate HDF5 files.
 - # Processed data will save in data/stanford_indoor3d/.

"""
#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import os, sys, pdb
from tqdm import tqdm
from indoor3d_util import DATA_PATH, collect_point_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/s3dis/anno_paths.txt'))]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]
# pdb.set_trace()

output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d')
if not os.path.exists(output_folder):
	os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
# https://github.com/charlesq34/pointnet/issues/45
for anno_path in tqdm(anno_paths):

	try:
		elements = anno_path.split('/')
		out_filename = elements[-3] + '_' + elements[-2] + '.npy'  # Area_1_hallway_1.npy
		collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
	except:
		print(anno_path, 'ERROR!!')
		pdb.set_trace()
