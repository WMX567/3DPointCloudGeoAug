import numpy as np
import torch
import os
import torch.utils.data as Data

from open3d import io

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
    p_inputs = torch.unsqueeze(torch.cat([x[2] for x in batch]), 0)
    p_inputs_adv = torch.cat([x[3] for x in batch], 1)
    pd = torch.cat([x[4] for x in batch], 1)
    gts = torch.stack([x[5] for x in batch])
    coarse_gts = torch.stack([x[6] for x in batch])
    return ids, npts, p_inputs, p_inputs_adv, pd, gts, coarse_gts


def get_item_detial(x1, x2, p_input_size, gt_size, num_coarse):

    p_input = resample_pcd(x1, p_input_size) if x1.shape[0] > p_input_size else x1

    npt = x1.shape[0] if x1.shape[0] < p_input_size else p_input_size

    gt = resample_pcd(x2, gt_size)

    choice = np.random.choice(len(gt), num_coarse, replace=True)
    coarse_gt = gt[choice, :]

    p_input = p_input.astype(np.float32)
    gt = gt.astype(np.float32)
    coarse_gt = coarse_gt.astype(np.float32)

    p_input.setflags(write=1)
    gt.setflags(write=1)
    coarse_gt.setflags(write=1)

    p_input = torch.from_numpy(p_input)
    gt = torch.from_numpy(gt)
    coarse_gt = torch.from_numpy(coarse_gt)

    return npt, p_input, gt, coarse_gt

class ShapeNet(Data.Dataset):

    def __init__(self, is_train, path, p_input_size, output_size, num_coarse):

        self.p_input_size = p_input_size
        self.num_coarse = num_coarse
        self.gt_size = output_size
        self.path = path
        if is_train:
            self.size = 231792
        else:
            self.size = 800

    def __getitem__(self, index):

        x1 = np.load(self.path+'partial/partial_'+str(index)+'.npy')
        x2 = np.load(self.path+'ground_truth/gt_'+str(index)+'.npy')

        npt, p_input, gt, coarse_gt = get_item_detial(x1, x2, self.p_input_size, self.gt_size, self.num_coarse)

        return index, npt, p_input, gt, coarse_gt

    def __len__(self):
        return self.size


class ShapeNetAdv(Data.Dataset):

    def __init__(self, is_train, path, path_adv, path_pd,
     p_input_size, output_size, num_coarse):

        self.p_input_size = p_input_size
        self.num_coarse = num_coarse
        self.gt_size = output_size
        self.path = path
        self.path_adv = path_adv
        self.path_pd = path_pd
        if is_train:
            self.size = 231792
        else:
            self.size = 800

    def __getitem__(self, index):

        x1 = np.load(self.path+'partial/partial_'+str(index)+'.npy')
        x2 = np.load(self.path+'ground_truth/gt_'+str(index)+'.npy')

        npt, p_input, gt, coarse_gt = get_item_detial(x1, x2, self.p_input_size, self.gt_size, self.num_coarse)

        p_input_adv = torch.load(self.path_adv + 'pd_'+str(index)+'.pt')
        pd = torch.load(self.path_pd+'pd_'+str(index)+'.pt')

        return index, npt, p_input, p_input_adv, pd, gt, coarse_gt

    def __len__(self):
        return self.size


class ShapeNetTest(Data.Dataset):

    def __init__(self, list_path, p_path, gt_path, p_input_size, output_size, num_coarse):

        self.p_input_size = p_input_size
        self.num_coarse = num_coarse
        self.gt_size = output_size
        self.p_path = p_path
        self.gt_path = gt_path
        self.file_list = list()

        with open(list_path, 'r') as f:
            self.file_list = [line.strip() for line in f]


    def __getitem__(self, index):

        filename = self.file_list[index]
        partial_input_path = os.path.join(self.p_path, filename+'.pcd')
        ground_truth_path = os.path.join(self.gt_path, filename+'.pcd')

        x1 = np.array(io.read_point_cloud(partial_input_path).points)
        x2 = np.array(io.read_point_cloud(ground_truth_path).points)

        category, _ = filename.split('/')
        npt, p_input, gt, coarse_gt = get_item_detial(x1, x2, self.p_input_size, self.gt_size, self.num_coarse)

        return category, npt, p_input, gt, coarse_gt


    def __len__(self):
        return len(self.file_list)


class ShapeNetAdvN(Data.Dataset):

    def __init__(self, is_train, path, path_adv, path_pd,
     p_input_size, output_size, num_coarse):

        self.p_input_size = p_input_size
        self.num_coarse = num_coarse
        self.gt_size = output_size
        self.path = path
        self.path_adv = path_adv
        self.path_pd = path_pd
        if is_train:
            self.size = 231792
        else:
            self.size = 800

    def __getitem__(self, index):

        x1 = np.load(self.path+'partial/partial_'+str(index)+'.npy')
        x2 = np.load(self.path+'ground_truth/gt_'+str(index)+'.npy')

        npt, p_input, gt, coarse_gt = get_item_detial(x1, x2, self.p_input_size, self.gt_size, self.num_coarse)

        p_input_adv = torch.load(self.path_adv + 'partial_'+str(index)+'.npy')
        n = torch.load(self.path_pd+'normal_'+str(index)+'.pt')

        return index, npt, p_input, p_input_adv, n, gt, coarse_gt

    def __len__(self):
        return self.size


# if __name__ == '__main__':

#     valid_path = 'valid.lmdb'
#     batch_size = 32
#     num_coarse = 1024
#     p_input_size = 2048
#     output_size = 16384

#     train_dataset = ShapeNet(False, valid p_input_size, output_size, num_coarse)
#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, 
#         shuffle=True, num_workers=4, collate_fn=collate_fn)

#     for i, data in enumerate(train_dataloader):
#          npts, p_inputs, gts, coarse_gts = data
#          print(p_inputs.size(), gts.size(), coarse_gts.size(), len(npts))
#          break

    
    
    
