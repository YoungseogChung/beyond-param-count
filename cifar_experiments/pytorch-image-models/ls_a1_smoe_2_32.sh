#!/bin/bash

sh distributed_train.sh 1 \
    --dataset torch/cifar100 \
    --dataset-download True \
    --model astroformer_1 \
    --num-classes 100 \
    --img-size 96 \
    --in-chans 3 \
    --input-size 3 96 96 \
    --batch-size 800 \
    --grad-accum-steps 1 \
    --opt adamw \
    --sched cosine \
    --lr-base 3e-4 \
    --lr-cycle-decay 1e-2 \
    --lr-k-decay 1 \
    --warmup-lr 1e-5 \
    --epochs 500 \
    --warmup-epochs 5 \
    --mixup 0.8 \
    --smoothing 0.1 \
    --drop 0.1 \
    --save-images \
    --amp \
    --amp-impl apex \
    --output smoe_astro \
    --log-wandb \
    --num-experts 2 \
    --moe-ratio 32 \
    --experiment ne2_mr32 \

