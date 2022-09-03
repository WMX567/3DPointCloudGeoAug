import os
import lmdb
import torch
import numpy as np
from tensorpack import dataflow


if __name__ == '__main__':

    lmdb_train = '/scratch/mw4355/shapenet/train.lmdb'
    df_train = dataflow.LMDBSerializer.load(lmdb_train, shuffle=False)

    for i, data in enumerate(df_train):
        np.save('/scratch/mw4355/shapenet/train_saved/partial/partial_'+str(i)+'.npy', data[1])
        np.save('/scratch/mw4355/shapenet/train_saved/ground_truth/gt_'+str(i)+'.npy', data[2])


    lmdb_valid = '/scratch/mw4355/shapenet/valid.lmdb'
    df_valid = dataflow.LMDBSerializer.load(lmdb_valid, shuffle=False)
    

    for i, data in enumerate(df_valid):
        np.save('/scratch/mw4355/shapenet/valid_saved/partial/partial_'+str(i)+'.npy', data[1])
        np.save('/scratch/mw4355/shapenet/valid_saved/ground_truth/gt_'+str(i)+'.npy', data[2])


        
