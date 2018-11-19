#!/bin/bash -login

#SBATCH --job-name=lstm-comparison-seqMNIST
#SBATCH --partition gpu
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=5:0:0
#SBATCH --mem=2500M

module add languages/anaconda3/5.2.0-tflow-1.11
module add libs/cuda/9.0-gcc-5.4.0-2.26
module add libs/cudnn/9.0-cuda-9.0

echo 'Comparing LSTM variants on sequential MNIST'

srun python benchmarks/seqMNIST/run.py  \
    --model subLSTM --nlayers 1 --nhid 50 \
    --lr 10e-4 --optim rmsprop --epochs 40 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 \
    --save ./benchmarks/seqMNIST/results

srun python benchmarks/seqMNIST/run.py  \
    --model fix-subLSTM --nlayers 1 --nhid 50 \
    --lr 10e-4 --optim rmsprop --epochs 40 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 \
    --save ./benchmarks/seqMNIST/results

srun python benchmarks/seqMNIST/run.py  \
    --model LSTM --nlayers 1 --nhid 50 \
    --lr 10e-4 --optim rmsprop --epochs 40 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 \
    --save ./benchmarks/seqMNIST/results

srun python benchmarks/seqMNIST/run.py  \
    --model GRU --nlayers 1 --nhid 50 \
    --lr 10e-4 --optim rmsprop --epochs 40 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 \
    --save ./benchmarks/seqMNIST/results