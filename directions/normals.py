import math
import os
import sys
import time
import argparse
import logging

from open3d import io
import numpy as np
import torch
from pytorch3d.ops import knn_points, knn_gather

from principal_directions import estimate_normal

if __name__ == '__main__':

	# logging.basicConfig(filename='myrun1.log', level=logging.INFO)
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--train_path', type=str, default='/scratch/mw4355/shapenet/train_saved/')
	# parser.add_argument('--save_dir', type=str, default='/scratch/mw4355/shapenet/normals/')
	# args = parser.parse_args()

	parser = argparse.ArgumentParser()
	parser.add_argument('--list_path', type=str, default='/scratch/mw4355/shapenet/test.list')
	parser.add_argument('--p_path', type=str, default='/scratch/mw4355/shapenet/test/partial/')
	parser.add_argument('--save_dir', type=str, default='/scratch/mw4355/shapenet/pd_see/')
	args = parser.parse_args()

	file_list = []
	with open(args.list_path, 'r') as f:
		file_list = [line.strip() for line in f]

		for filename in file_list:
			category, _ = filename.split('/')
			if category == "03001627":
				partial_input_path = os.path.join(args.p_path, filename+'.pcd')
				p_input = np.array(io.read_point_cloud(partial_input_path).points)
				p_input.setflags(write=1)
				p_input = torch.from_numpy(p_input).unsqueeze(0).float()
				normal = estimate_normal(p_input.permute(0,2,1), 20)
				torch.save(normal, args.save_dir+category+'_normal.pt')
				break
