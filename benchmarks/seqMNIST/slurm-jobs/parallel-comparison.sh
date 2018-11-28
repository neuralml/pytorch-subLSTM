#!/bin/bash -login

#SBATCH --job-name=lstm-comparison-seqMNIST
#SBATCH --partition gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=5:0:0
#SBATCH --mem=16000M

module add languages/anaconda3/5.2.0-tflow-1.11
module add libs/cuda/9.0-gcc-5.4.0-2.26
module add libs/cudnn/9.0-cuda-9.0

# Put the module on PYTHON_PATH since I cannot install it on the BC4 cluster
export PYTHONPATH=$PYTHONPATH:../../src/

echo 'Comparing LSTM variants on sequential MNIST'

srun -N 1 python benchmarks/seqMNIST/run.py  \
    --model subLSTM --nlayers 1 --nhid 50 \
    --optim rmsprop --lr 10e-4 --epochs 40 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 \
    --save ./benchmarks/seqMNIST/results &

srun -N 1 python benchmarks/seqMNIST/run.py  \
    --model fix-subLSTM --nlayers 1 --nhid 50 \
    --optim rmsprop --lr 10e-4 --epochs 40 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 \
    --save ./benchmarks/seqMNIST/results &

srun -N 1 python benchmarks/seqMNIST/run.py  \
    --model LSTM --nlayers 1 --nhid 50 \
    --optim rmsprop --lr 10e-4 --epochs 40 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 \
    --save ./benchmarks/seqMNIST/results &

srun -N 1 python benchmarks/seqMNIST/run.py  \
    --model GRU --nlayers 1 --nhid 50 \
    --optim rmsprop --lr 10e-4 --epochs 40 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 \
    --save ./benchmarks/seqMNIST/results &
wait