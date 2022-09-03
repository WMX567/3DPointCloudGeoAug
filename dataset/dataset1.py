import torch.utils.data as data
import numpy as np
import os, sys
from open3d import io
import random
import os
import json
import torch

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

def collate_fn(batch):
    ids = np.stack([x[0] for x in batch])
    npts = np.stack([x[1] for x in batch])
    npts = tuple(npts)
    p_inputs = [x[2] for x in batch]
    p_inputs = torch.unsqueeze(torch.cat([x for x in p_inputs]), 0)
    gts = torch.stack([x[3] for x in batch])
    coarse_gts = torch.stack([x[4] for x in batch])
    return ids, npts, p_inputs, gts, coarse_gts


def collate_fn_adv(batch):
    ids = np.stack([x[0] for x in batch])
    npts = np.stack([x[1] for x in batch])
    npts = tuple(npts)
    p_inputs = [x[2] for x in batch]
    p_inputs = torch.unsqueeze(torch.cat([x for x in p_inputs]), 0)
    p_inputs_adv = torch.cat([x[3] for x in batch],2)
    pd = torch.cat([x[4] for x in batch],1)
    gts = torch.stack([x[5] for x in batch])
    coarse_gts = torch.stack([x[6] for x in batch])
    return ids, npts, p_inputs, p_inputs_adv, pd, gts, coarse_gts

class ShapeNet(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, subset, ncoarse, is_adv):

        self.data_path = '/scratch/mw4355/ShapeNetCompletion/'
        self.subset = subset
        self.is_adv = is_adv
        self.n_coarse = ncoarse
        self.n_points = 2048

        # Load the dataset indexing file
        self.dataset_categories = []
        with open('/scratch/mw4355/3D-Compeletion/dataset/data/PCN.json', 'r') as f:
            self.dataset_categories = json.loads(f.read())

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)


    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            samples = dc[subset]
            for s in samples:
                file_list.append({
                    'taxonomy_id':dc['taxonomy_id'],
                    'model_id':s,
                    'partial_path': [self.data_path + subset + '/partial/' + dc['taxonomy_id']+'/'+s+'/0'+str(i)+'.pcd' for i in range(n_renderings)],
                    'adv_partial_path' : [self.data_path + 'adv_samples/' + s + '_' + str(i+1)+'.npy' for i in range(n_renderings)],
                    'pd_path' : ['/scratch/mw4355/pd_mean/'+ s + '_' + str(i+1)+'.npy' for i in range(n_renderings)],
                    'gt_path': self.data_path + subset + '/complete/' + dc['taxonomy_id'] + '/'+ s + '.pcd'
                })

        return file_list


    def __getitem__(self, idx):

        sample = self.file_list[idx]
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        partial = np.array(io.read_point_cloud(sample['partial_path'][rand_idx]).points).astype(np.float32)
        if partial.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - partial.shape[0], 3))
            partial = np.concatenate([partial, zeros])

        npt = partial.shape[0]

        gt = np.array(io.read_point_cloud(sample['gt_path']).points).astype(np.float32)

        choice = np.random.choice(len(gt), self.n_coarse, replace=True)
        coarse_gt = gt[choice, :]

        partial = torch.from_numpy(partial).float()
        gt = torch.from_numpy(gt).float()
        coarse_gt = torch.from_numpy(coarse_gt).float()

        if self.is_adv:
            partial_adv = np.load(sample['adv_partial_path'][rand_idx]).astype(np.float32)
            partial_adv = torch.from_numpy(partial_adv).float()
            pd = np.load(sample['pd_path'][rand_idx]).astype(np.float32)
            pd = torch.from_numpy(pd).float()
            return sample['adv_partial_path'][rand_idx], npt, partial, partial_adv, pd, gt, coarse_gt

        return sample['taxonomy_id'], npt, partial, gt, coarse_gt

    def __len__(self):
        return len(self.file_list)

if __name__ == '__main__':
    ShapeNet('train', False)

