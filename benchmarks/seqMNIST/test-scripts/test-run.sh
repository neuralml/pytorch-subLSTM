#!/bin/bash

cd ..
source ../../.venv/bin/activate

python run.py --model subLSTM --nlayers 1 --nhid 50 \
    --optim adam --lr 1e-3 --l2-norm 0.0001 --epochs 2 --batch-size 50 \
    --seed 18092 --cuda --log-interval 50 --data MNIST --save results \
    --input-size 28 --track-hidden --verbose