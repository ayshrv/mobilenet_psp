#!/bin/bash

python train.py \
  --data_list=list/train_extra_list.txt \
  --log_dir=logs/train6 \
  --gpu=0 \
  --num_epochs=15 \
  --decay_steps=5 \
  --optimizer=momentum \
  --update_mean_var=True \
  --update_beta=True

python train_continue.py \
  --data_list=list/train_extra_list.txt \
  --pretrained_checkpoint=logs/train6/checkpoints/model.ckpt-15 \
  --log_dir=logs/train6 \
  --gpu=0 \
  --num_epochs=15 \
  --start_epoch=16 \
  --update_mean_var=False \
  --update_beta=True

python train_continue.py \
    --data_list=list/train_extra_list.txt \
    --pretrained_checkpoint=logs/train6/checkpoints/model.ckpt-30 \
    --log_dir=logs/train6 \
    --gpu=0 \
    --num_epochs=30 \
    --start_epoch=31 \
    --update_mean_var=False \
    --update_beta=False
