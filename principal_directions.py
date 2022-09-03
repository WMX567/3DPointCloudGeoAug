import math
import os
import sys
import time
import argparse
import logging
import json

import numpy as np
import torch
from pytorch3d.ops import knn_points, knn_gather
from open3d import io

logging.basicConfig(filename='data_pd.log', level=logging.INFO)
# This function is from: 
# https://github.com/Gorilla-Lab-SCUT/GeoA3/blob/master/Lib/utility.py
def estimate_normal(pc, k):
    # pc : [b, 3, n]
    b,_,n=pc.size()
    # get knn point set matrix
    inter_KNN = knn_points(pc.permute(0,2,1), pc.permute(0,2,1), K=k+1) # [dists:[b,n,k+1], idx:[b,n,k+1]]
    nnpts = knn_gather(pc.permute(0,2,1), inter_KNN.idx).permute(0,3,1,2)[:,:,:,1:].contiguous() # [b, 3, n ,k]

    # get covariance matrix and smallest eig-vector of individual point
    normal_vector = []

    for i in range(b):
      
        curr_point_set = nnpts[i].detach().permute(1,0,2) #curr_point_set:[n, 3, k]
        curr_point_set_mean = torch.mean(curr_point_set, dim=2, keepdim=True) #curr_point_set_mean:[n, 3, 1]
        curr_point_set = curr_point_set - curr_point_set_mean #curr_point_set:[n, 3, k]
        curr_point_set_t = curr_point_set.permute(0,2,1) #curr_point_set_t:[n, k, 3]
        fact = 1.0 / (k-1)
        cov_mat = fact * torch.bmm(curr_point_set, curr_point_set_t) #cov_mat:[n, 3, 3]
        eigenvalue, eigenvector=torch.symeig(cov_mat, eigenvectors=True)    #eigenvalue:[n, 3], eigenvector:[n, 3, 3]
        persample_normal_vector = torch.gather(eigenvector, 2, torch.argmin(eigenvalue, dim=1).unsqueeze(1).unsqueeze(2).expand(n, 3, 1)).squeeze() #persample_normal_vector:[n, 3]

        #recorrect the direction via neighbour direction
        nbr_sum = curr_point_set.sum(dim=2)  #curr_point_set:[n, 3]
        sign_cov_mat = torch.bmm(persample_normal_vector.view(n, 1, 3), nbr_sum.view(n, 3, 1))
        sign = -torch.sign(sign_cov_mat).squeeze(2)
        sign[sign==0] = 1.0
        persample_normal_vector = sign * persample_normal_vector
        normal_vector.append(persample_normal_vector)

    normal_vector = torch.stack(normal_vector, 0) #normal_vector:[b, n, 3]

    return normal_vector


def get_angle(d1, d2, n, near_normals):

	n1, n2 = None, None
	m,k,_ = near_normals.size()

	d1 = d1.unsqueeze(2).expand(m,3,k).permute(0,2,1).contiguous()
	angles = torch.arccos(torch.einsum('bij,bij->bi', near_normals, d1))
	sorted_angles, indices = torch.sort(angles)
	
	near1 = torch.gather(near_normals, dim=1, index=indices[:,-1].unsqueeze(1).unsqueeze(2).expand(m,1,3)).squeeze(1)
	near2 = torch.gather(near_normals, dim=1, index=indices[:,-2].unsqueeze(1).unsqueeze(2).expand(m,1,3)).squeeze(1)
	# normal vector for the first sector
	v1 = torch.cross(near1, near2, dim=1)
	n1 = torch.cross(v1, d2, dim=1)
	n1 /= torch.norm(n1, dim=1).unsqueeze(1).expand(m, 3)
	# Make sure n1 is at same half space as n
	test_d = torch.einsum('bi,bi->b', n1, n).unsqueeze(1).expand(m,3)
	n1 = torch.where(test_d < 0.0, -n1, n1)

	near3 = torch.gather(near_normals, dim=1, index=indices[:,0].unsqueeze(1).unsqueeze(2).expand(m,1,3)).squeeze(1)
	near4 = torch.gather(near_normals, dim=1, index=indices[:,1].unsqueeze(1).unsqueeze(2).expand(m,1,3)).squeeze(1)
	# normal vector for the second sector
	v2 = torch.cross(near3, near4, dim=1)
	n2 = torch.cross(v2, d2, dim=1) 
	n2 /= torch.norm(n2, dim=1).unsqueeze(1).expand(m, 3)
	# Make sure n2 is at same half space as n
	test_d = torch.einsum('bi,bi->b', n2, n).unsqueeze(1).expand(m,3)
	n2 = torch.where(test_d < 0.0, -n2, n2)

	return torch.arccos(torch.einsum('bi,bi->b', n1, n2))


def fast_get_initial(normals,c, s, m):
	D1 = normals.clone()
	D2 = normals.clone()
	z = torch.zeros(m)
	D1[:,0], D1[:,1], D1[:,2] = normals[:,1], -normals[:,0], z
	D2[:,0], D2[:,1], D2[:,2] = z, -normals[:,2], normals[:,1]
	D_i = torch.where(normals[:,2].unsqueeze(1).expand(m,3) < normals[:,0].unsqueeze(1).expand(m,3), D1, D2)

	R = torch.einsum('bi,bj->bij', (normals, normals))*(1-c)
	C = torch.tensor([[c,0,0],[0,c,0],[0,0,c]]).unsqueeze(0).repeat(m,1,1)
	T = torch.zeros([m,3,3])
	T[:,0,1], T[:,0,2] = -normals[:,2], normals[:,1]
	T[:,1,0], T[:,1,2] = normals[:,2],  -normals[:,0]
	T[:,2,0], T[:,2,1] = -normals[:,1], normals[:,0]
	T *= s
	R += T + C
	return D_i, R


def find_direction(pc, normal_k, Di_k, nn_k):

	# Get Normals and Neighbours
	pc = pc
	normals = estimate_normal(pc, normal_k) # noraml: [b,3,n]
	inter_KNN = knn_points(pc.permute(0,2,1), pc.permute(0,2,1), K=nn_k+1) # [dists:[b,n,k+1], idx:[b,n,k+1]]
	neighbour_normals = knn_gather(normals, inter_KNN.idx[:,:,1:])

	b,m,_= normals.size()

	# The angle rotation every time
	theta = np.pi / Di_k
	c, s = math.cos(theta), math.sin(theta)

	directions = []
	for i in range(b):

		D_i, R = fast_get_initial(normals[i],c, s, m)
		near_normals = neighbour_normals[i]
		min_direction = D_i
		min_angle = None

		for k in range(Di_k):
			if k >= 1:
				D_i = torch.bmm(R, D_i.unsqueeze(2)).squeeze(2)

			D_i_ = torch.cross(normals[i], D_i, dim=1)
			D_i /= torch.norm(D_i, dim=1).unsqueeze(1).expand(m, 3)
			D_i_ /= torch.norm(D_i_, dim=1).unsqueeze(1).expand(m, 3)

			# Get alpha, beta
			angle = get_angle(D_i, D_i_, normals[i], near_normals)
			# beta = get_angle(D_i_, D_i, normals[i], near_normals)

			#min_dir_temp = torch.where(alpha.unsqueeze(1).expand(m,3) < beta.unsqueeze(1).expand(m,3), D_i, D_i_)

			if k == 0:
				#max_diff = torch.abs(beta-alpha).unsqueeze(1).expand(m,3)
				min_angle = torch.abs(angle).unsqueeze(1).expand(m,3)
				# min_direction = min_dir_temp
				# min_direction /= torch.norm(min_direction, dim=1).unsqueeze(1).expand(m, 3)
				min_direction = D_i
				min_direction /= torch.norm(min_direction, dim=1).unsqueeze(1).expand(m, 3)
				continue

			#temp_max_diff = torch.abs(beta-alpha).unsqueeze(1).expand(m,3)
			temp_angle = torch.abs(angle).unsqueeze(1).expand(m,3)
			min_direction = torch.where(temp_angle < min_angle, D_i, min_direction)
			min_direction /= torch.norm(min_direction, dim=1).unsqueeze(1).expand(m, 3)
			min_angle = torch.where(temp_angle < min_angle, temp_angle, min_angle)
			# mean_direction = torch.where(temp_max_diff > max_diff, D_i + D_i_, mean_direction)
			# mean_direction /= torch.norm(mean_direction, dim=1).unsqueeze(1).expand(m, 3)
			# max_diff = torch.where(temp_max_diff > max_diff, temp_max_diff, max_diff)
		
		directions.append(min_direction)

	return torch.stack(directions,0)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_path', type=str, default='/scratch/mw4355/shapenet/train_saved/')
	parser.add_argument('--save_dir', type=str, default='/scratch/mw4355/shapenet/min_dir/')
	args = parser.parse_args()

	for i in range(231792):
		logging.info("#: "+ str(i))
		x1 = np.load(args.train_path+'partial/partial_'+str(i)+'.npy')
		x1.setflags(write=1)
		x1 = torch.from_numpy(x1).unsqueeze(0).float()
		pd = find_direction(x1.permute(0,2,1), 20, 36, 10)
		torch.save(pd,args.save_dir + 'pd_'+str(i)+'.pt')
	# x1 = torch.rand(1,6,3).float()
	# pd = find_direction(x1.permute(0,2,1), 20, 36, 10)
	# print(pd.size())





