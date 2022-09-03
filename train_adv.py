import argparse
import os
import logging
import time

import torch
import torch.optim as optim
import torch.multiprocessing as mp

import numpy as np
import open3d as o3d
import random
from model_with_extra_bn import AutoEncoderBN
from loss import ChamferDistance, EarthMoverDistance

from dataset.dataset import ShapeNet, ShapeNetAdv, collate_fn, collate_fn_adv

from attack_pd import RepresentationAdv

# Logging File Created
logging.basicConfig(filename='myrun1.log', level=logging.INFO)

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt

def train_(local_rank, args):

    print("GPU Number: ", local_rank)

    # Configuration for Training
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Set device，GPU Communication: NCCL
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl',init_method='env://',
        world_size=args.gpus, rank=local_rank)
    device = torch.device(f'cuda:{local_rank}')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # make it train faster
    torch.backends.cudnn.deterministic = True

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    # Random Setting
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Dataset
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++

    args.batch_size = int(args.batch_size / args.gpus)

    train_dataset = ShapeNetAdv(True, args.train_path, args.adv_path, args.pd_path, args.num_input, 
        args.num_dense, args.num_coarse)

    evens = list(range(0, len(train_dataset), 2))
    odds = list(range(1, len(train_dataset), 2))
    trainset_1 = torch.utils.data.Subset(train_dataset, evens)
    trainset_2 = torch.utils.data.Subset(train_dataset, odds)

    train_sampler_evens = torch.utils.data.distributed.DistributedSampler(trainset_1)
    train_sampler_odds = torch.utils.data.distributed.DistributedSampler(trainset_2)

    train_dataloader_evens = torch.utils.data.DataLoader(trainset_1, args.batch_size, 
            pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn_adv, sampler=train_sampler_evens)
    train_dataloader_odds = torch.utils.data.DataLoader(trainset_2, args.batch_size, 
            pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn_adv, sampler=train_sampler_odds)

    valid_dataset = ShapeNet(False, args.valid_path, args.num_input, args.num_dense, args.num_coarse)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, args.batch_size, 
            pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn, sampler=valid_sampler)

    if local_rank == 0:
        logging.info("Trian Even Length: %d", len(train_dataloader_evens))
        logging.info("Trian Odd Length: %d", len(train_dataloader_odds))

    # Loss
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++

    cd_loss = ChamferDistance().to(device)
    emd_loss = EarthMoverDistance().to(device)
    loss_d1 = cd_loss if args.loss_d1 == 'cd' else emd_loss
    loss_d2 = cd_loss

    # Model and Optimizer
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    network = AutoEncoderBN()
    network.cuda(local_rank)
    network= torch.nn.parallel.DistributedDataParallel(network, device_ids=[local_rank],find_unused_parameters=True)
    if args.model is not None:
        logging.info('Loaded trained model from {}.'.format(args.model))
        torch.distributed.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        network.load_state_dict(torch.load(args.model,map_location=map_location))
    else:
        logging.info('Begin training new model.')

    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # Adversarial Initialization
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Rep = RepresentationAdv(args.epsilon, args.beta, args.k, _type='linf')

    # Basic Parameters & Helper function
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    minimum_loss = 1e4
    best_epoch = 0

    # Running
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for epoch in range(1, args.epochs + 1):

        if epoch % 2 == 1:
            train_dataloader = train_dataloader_odds
        else:
            train_dataloader = train_dataloader_evens

        train_sampler_odds.set_epoch(epoch)
        train_sampler_evens.set_epoch(epoch)

        # training
        network.train()

        total_loss, iter_count = 0, 0

        if local_rank == 0:
            logging.info('======================')
            logging.info('Adv:'+ str(args.adv))

        s = time.process_time() 
        for data in train_dataloader:

            ids, npts, partial_input, partial_input_adv, pd, dense_gt, coarse_gt = data

            coarse_gt = coarse_gt.cuda(local_rank, non_blocking=True)
            dense_gt = dense_gt.cuda(local_rank, non_blocking=True)
            partial_input = partial_input.cuda(local_rank, non_blocking=True)

            partial_input = partial_input.permute(0, 2, 1)
            partial_input_adv = partial_input_adv.permute(0,2,1)

            if args.adv:
                if epoch % 30 == 1 or epoch % 30 == 2:
                    partial_input_adv = partial_input

                partial_input_adv = partial_input_adv.cuda(local_rank, non_blocking=True)
                pd = pd.cuda(local_rank, non_blocking=True)
    
                new_input_adv = Rep.get_loss(npts, partial_input_adv, coarse_gt, dense_gt, pd, 
            network, loss_d1, random_start=False)

                temp = torch.split(new_input_adv, npts, dim=2)
                for i, index in enumerate(ids):
                    torch.save(temp[i].permute(0,2,1).cpu(), args.adv_path + 'pd_'+str(index)+'.pt')

                partial_input_adv = new_input_adv

            # Normal Samples          
            v, y_coarse, y_detail = network(partial_input, npts, False)

            y_coarse = y_coarse.permute(0, 2, 1)
            y_detail = y_detail.permute(0, 2, 1)
            
            loss = loss_d1(coarse_gt, y_coarse) + args.alpha * loss_d2(dense_gt, y_detail)

            torch.distributed.barrier()

            reduced_loss = reduce_tensor(loss.data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Adv Samples
            if args.adv:

                v_adv, y_coarse_adv, y_detail_adv = network(partial_input_adv, npts, True)

                y_coarse_adv = y_coarse_adv.permute(0, 2, 1)
                y_detail_adv = y_detail_adv.permute(0, 2, 1)

                loss_adv = args.adv_alpha * (loss_d1(coarse_gt, y_coarse_adv) + args.alpha * loss_d2(dense_gt, y_detail_adv))

                torch.distributed.barrier()

                optimizer.zero_grad()
                loss_adv.backward()
                optimizer.step()

                reduced_loss += reduce_tensor(loss_adv.data)
    
            iter_count += 1
            total_loss += reduced_loss.item()

            if iter_count % 100 == 0 and local_rank == 0:
                logging.info("100 Batch Time:"+str(time.process_time()-s))
                s = time.process_time() 
                logging.info("Training epoch {}/{}, iteration {}/{}: loss is {}".format(epoch, args.epochs, iter_count, len(train_dataloader), loss.item()))
        
        scheduler.step()

        if local_rank == 0:
            logging.info("\033[31mTraining epoch {}/{}: avg loss = {}\033[0m".format(epoch, args.epochs, total_loss / iter_count))

        # Evaluation
        network.eval()
        with torch.no_grad():
            total_loss, iter_count = 0, 0
            for data in valid_dataloader:
                
                ids, npts, partial_input, dense_gt, coarse_gt = data

                partial_input = partial_input.cuda(local_rank, non_blocking=True)
                coarse_gt = coarse_gt.cuda(local_rank, non_blocking=True)
                dense_gt = dense_gt.cuda(local_rank, non_blocking=True)

                partial_input = partial_input.permute(0, 2, 1)
                
                v, y_coarse, y_detail = network(partial_input, npts, False)

                y_coarse = y_coarse.permute(0, 2, 1)
                y_detail = y_detail.permute(0, 2, 1)

                loss = loss_d1(coarse_gt, y_coarse) + args.alpha * loss_d2(dense_gt, y_detail)

                torch.distributed.barrier()

                reduced_loss = reduce_tensor(loss.data)
                total_loss += reduced_loss.item()
                iter_count += 1

            mean_loss = total_loss / iter_count

            if local_rank == 0:
                logging.info("\033[31mValidation epoch {}/{}, loss is {}\033[0m".format(epoch, args.epochs, mean_loss))

            # records the best model and epoch
            if mean_loss < minimum_loss and local_rank == 0:
                best_epoch = epoch
                minimum_loss = mean_loss
                torch.save(network.state_dict(), args.log_dir + args.save_name)

        if local_rank == 0:
            logging.info("\033[31mBest model (lowest loss) in epoch {}\033[0m".format(best_epoch))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='/scratch/mw4355/shapenet/train_saved/')
    parser.add_argument('--valid_path', type=str, default='/scratch/mw4355/shapenet/valid_saved/')
    parser.add_argument('--adv_path', type=str, default='/scratch/mw4355/shapenet/adv_saved/min_dir/')
    parser.add_argument('--pd_path', type=str, default='/scratch/mw4355/shapenet/min_dir/')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--adv', type=int, default=1)
    parser.add_argument('--num_input', type=int, default=2048)
    parser.add_argument('--num_coarse', type=int, default=1024)
    parser.add_argument('--num_dense', type=int, default=16384)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--adv_alpha', type=float, default=1.0)
    parser.add_argument('--loss_d1', type=str, default='cd')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='/scratch/mw4355/models_saved/')
    parser.add_argument('--save_name', type=str, default='adv_bn_min_abs.pth')
    parser.add_argument('--epsilon', type=float, default=0.0314,
            help='maximum perturbation of adversaries (8/255 for cifar-10)')
    parser.add_argument('--beta', type=float, default=0.7,
            help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
    parser.add_argument('--k', type=int, default=1,
            help='maximum iteration when generating adversarial examples')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(train_, nprocs=args.gpus,args=(args,))

if __name__ == '__main__':

    main()
