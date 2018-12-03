#!/bin/bash

cd ..
source activate torch

export PYTHONPATH=$PYTHONPATH:../../src/ 

python -O run.py --model subLSTM --nlayers 1 --nhid 50 \
    --seq-length 20 --num-addends 2 --min-arg 0 --max-arg 50 \
    --training-size 2500 --testing-size 1000 --batch-size 50 \
    --optim rmsprop --lr 1e-4 --l2-norm 0.0 --clip 1 --epochs 1000 \
    --seed 18092 --cuda --log-interval 5 --save results \
    --track-hidden --verbose
