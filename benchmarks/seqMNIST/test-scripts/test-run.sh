#!/bin/bash

cd ..
source activate torch

export PYTHONPATH=$PYTHONPATH:../../src/ 

python run.py --model subLSTM --nlayers 1 --nhid 100 \
    --optim rmsprop --lr 1e-3 --l2-norm 0.0001 --epochs 1 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 --data MNIST --save results \
    --input-size 28 --track-hidden --verbose