#!/bin/bash

DATASET_DIR=~/SS/Datasets/cityscapes-images/

python train.py --data_dir=${DATASET_DIR} --log_dir=logs/train1 --gpu=0 --num_epochs=80 --optimizer=momentum --update_mean_var=True --update_beta=True

python train_continue.py --data_dir=${DATASET_DIR} --pretrained_checkpoint=logs/train1/checkpoints/model.ckpt-80 --log_dir=logs/train1 --gpu=0 --num_epochs=80 --start_epoch=81 --update_mean_var=False --update_beta=True

python train_continue.py --data_dir=${DATASET_DIR} --pretrained_checkpoint=logs/train1/checkpoints/model.ckpt-160 --log_dir=logs/train1 --gpu=0 --num_epochs=100 --start_epoch=161 --update_mean_var=False --update_beta=False
