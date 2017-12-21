#!/bin/bash

python train.py --log_dir=train5-Fine-Full-Momentum --gpu=1 --num_epochs=80 --optimizer=momentum --update_mean_var=True --update_beta=True

python train_continue.py --pretrained_checkpoint=logs/train5-Fine-Full-Momentum/checkpoints/model.ckpt-80 --log_dir=train5-Fine-Full-Momentum --gpu=1 --num_epochs=80 --start_epoch=81 --update_mean_var=False --update_beta=True

python train_continue.py --pretrained_checkpoint=logs/train5-Fine-Full-Momentum/checkpoints/model.ckpt-160 --log_dir=train5-Fine-Full-Momentum --gpu=1 --num_epochs=100 --start_epoch=161 --update_mean_var=False --update_beta=False
