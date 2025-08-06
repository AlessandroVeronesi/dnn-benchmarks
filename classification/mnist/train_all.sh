#!/bin/bash
# ------------------------------------------------------------------
# [Author] Robert Alexander Limas Sierra
#          Basic scripting code for training several IA models
# ------------------

models=(\
    "lenet"\
    "alexnet"\
    "vgg11"\
    "vgg19"\
    )

for model in "${models[@]}";
do
    python train.py --model $model --epochs 10 --batchsize 128
done
