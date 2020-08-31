#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import numpy as np

"""
	================================================
	=== Library for Point Cloud Utility Function ===
	================================================
	
	adapted from https://github.com/facebookresearch/hgnn/blob/master/utils/EarlyStoppingCriterion.py
	Arguments:
		patience (int): The maximum number of epochs with no improvement before early stopping should take place
		mode (str, can only be 'max' or 'min'): To take the maximum or minimum of the score for optimization
		min_delta (float, optional): Minimum change in the score to qualify as an improvement (default: 0.0)
	"""


def pc_normalize(pc):
	centroid = np.mean(pc, axis=0)
	pc = pc - centroid
	m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
	pc = pc / m
	return pc


def farthest_point_sample(point, npoint):
	"""
	Input:
		xyz: point cloud data, [N, D]
		npoint: number of samples
	Return:
		centroids: sampled point cloud index, [npoint, D]
	"""
	N, D = point.shape
	xyz = point[:, :3]
	centroids = np.zeros((npoint,))
	distance = np.ones((N,)) * 1e10
	farthest = np.random.randint(0, N)
	for i in range(npoint):
		centroids[i] = farthest
		centroid = xyz[farthest, :]
		dist = np.sum((xyz - centroid) ** 2, -1)
		mask = dist < distance
		distance[mask] = dist[mask]
		farthest = np.argmax(distance, -1)
	point = point[centroids.astype(np.int32)]
	return point


def random_shift_point_cloud(batch_data, shift_range=0.1):
	""" Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
	B, N, C = batch_data.shape
	shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
	for batch_index in range(B):
		batch_data[batch_index, :, :] += shifts[batch_index, :]
	return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
	""" Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
	B, N, C = batch_data.shape
	scales = np.random.uniform(scale_low, scale_high, B)
	for batch_index in range(B):
		batch_data[batch_index, :, :] *= scales[batch_index]
	return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
	""" batch_pc: BxNx3 """
	for b in range(batch_pc.shape[0]):
		# np.random.random() -> Return random floats in the half-open interval [0.0, 1.0).
		dropout_ratio = np.random.random() * max_dropout_ratio  # 0 ~ 0.875
		drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
		if len(drop_idx) > 0:
			batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
	return batch_pc


def translate_pointcloud_dgcnn(pointcloud):
	xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
	xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

	translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
	return translated_pointcloud


def jitter_pointcloud_dgcnn(pointcloud, sigma=0.01, clip=0.02):
	N, C = pointcloud.shape
	pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
	return pointcloud
