#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=300GB
#SBATCH --job-name=pcn_run
#SBATCH --mail-type=END
#SBATCH --mail-user=mw4355p@nyu.edu
#SBATCH --output=slurm_%j.out
##SBATCH -p nvidia

module purge

singularity exec --nv \
	    --overlay /scratch/mw4355/env/overlay-15GB-500K.ext3 \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; conda activate run; python train_adv.py -n 1 -g 4 -nr 0 --adv 1"

























