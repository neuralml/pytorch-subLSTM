#!/bin/bash -login

#SBATCH --job-name=lsmt-comparison-seqMNIST
#SBATCH --partition=gpu
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --time=7:0:0
#SBATCH --mem=2500M

module add languages/anaconda3/5.0.2-tflow-1.11
module add languages

echo 'Comparing LSTM variants on sequential MNIST'

srun python benchmarks/seqMNIST/main.py  \
    --model subLSMT --nlayers 1 --nhid 50 
    --lr 10e-4 --optim rmsrop --epochs 40 --batch-size 200 \
    --seed 18092 --cuda --log-interval 200 \
    --save ./benchmarks/seqMNIST/results

srun python benchmarks/seqMNIST/main.py  \
    --model fix-subLSMT --nlayers 1 --nhid 50 
    --lr 10e-4 --optim rmsrop --epochs 40 --batch-size 200 \
    --seed 18092 --cuda --log-interval 200 \
    --save ./benchmarks/seqMNIST/results

srun python benchmarks/seqMNIST/main.py  \
    --model LSMT --nlayers 1 --nhid 50 
    --lr 10e-4 --optim rmsrop --epochs 40 --batch-size 200 \
    --seed 18092 --cuda --log-interval 200 \
    --save ./benchmarks/seqMNIST/results

srun python benchmarks/seqMNIST/main.py  \
    --model GRU --nlayers 1 --nhid 50 
    --lr 10e-4 --optim rmsrop --epochs 40 --batch-size 200 \
    --seed 18092 --cuda --log-interval 200 \
    --save ./benchmarks/seqMNIST/results