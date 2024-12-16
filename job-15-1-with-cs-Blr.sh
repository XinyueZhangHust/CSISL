#!/bin/sh
#BATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --comment=txzqxm
hostname
source /home/xxxx/.bashrc
source activate CSISL
###module load app/cuda/11.6

cd tools
srun python -u script_train.py 15-1 0,1,2,3,4,5 0 --batch_size 8 --val_batch_size 8 --freeze_low --lr 0.0001  --mem_size 100 --conloss_sparsity --conloss_compression --KDLoss --KDLoss_prelogit  --name swin_voc2012_best --unknown --dataset voc --test_only
