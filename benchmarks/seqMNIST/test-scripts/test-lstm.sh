#!/bin/bash

cd ..
source activate torch

export PYTHONPATH=$PYTHONPATH:../../src/ 

python run.py --model LSTM --nlayers 1 --nhid 50 \
    --optim rmsprop --lr 10e-4 --l2-norm 0.0 --epochs 40 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 --data MNIST --save results
