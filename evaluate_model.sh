#!/bin/bash

# python ~/SS/mobilenet_psp/evaluate.py --gpu=1 --checkpoint_path=checkpoints/model.ckpt-43

# Method 1
DATASET_DIR=~/SS/Datasets/cityscapes-images/
DATA_LIST=list/eval_list.txt
CHECKPOINT_FOLDER=logs/train1-Fine-Full-Momentum/ #The folder which contains the checkpoints
EPOCH=43 #Which epoch to evaluate
GPU=1 #Which GPU to use

EVALUATE_LOG_FILE=$CHECKPOINT_FOLDER'evaluate.log'

CHECKPOINT_FILE=$CHECKPOINT_FOLDER'checkpoints/model.ckpt-'
CHECKPOINT_FILE+=$EPOCH

python evaluate.py \
  --data_dir=${DATASET_DIR} \
  --data_list=${DATA_LIST} \
  --checkpoint_path=${CHECKPOINT_FILE} \
  --evaluate_log_file=${EVALUATE_LOG_FILE} \
  --gpu=${GPU}
