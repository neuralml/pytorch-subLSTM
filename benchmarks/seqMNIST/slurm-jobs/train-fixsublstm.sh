#!/bin/bash -login

#SBATCH --job-name=gpu-test
#SBATCH --partition gpu_veryshort
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=1:0:0
#SBATCH --mem=16000M

module add languages/anaconda3/5.2.0-tflow-1.11
module add libs/cuda/9.0-gcc-5.4.0-2.26
module add libs/cudnn/9.0-cuda-9.0

cd ..
export PYTHONPATH=$PYTHONPATH:../../src/

echo 'Testing GPU using subLSTM'

srun python run.py  \
    --model fix-subLSTM --nlayers 1 --nhid 50 \
    --optim rmsprop --lr 1e-4 --l2-norm 0.1  \
    --epochs 100 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 \
    --save ./benchmarks/seqMNIST/results
