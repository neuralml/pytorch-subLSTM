#!/bin/bash -login

#SBATCH --job-name=lstm-comparison-seqMNIST
#SBATCH --partition gpu
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=10:0:0
#SBATCH --mem=2500M

module add languages/anaconda3/5.0.2-tflow-1.11
module add libs/cuda/9.0-gcc-5.4.0-2.26
module add libs/cudnn/9.0-cuda-9.0

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