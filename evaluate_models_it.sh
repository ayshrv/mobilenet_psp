#!/bin/bash

#Method 3
DATASET_DIR=~/SS/Datasets/cityscapes-images/
DATA_LIST=list/eval_list.txt
CHECKPOINT_FOLDER=logs/train1-Fine-Full-Momentum/ #The folder which contains the checkpoints
EVALUATE_LOG_FILE=$CHECKPOINT_FOLDER'evaluate.log'
GPU=1

#Used to evaluate all models from epoch 1 to 5 i.e. model.ckpt-1 to model.ckpt-5
START=1
END=5

for ((i=START;i<=END;i++)); do
  CHECKPOINT_FILE=$CHECKPOINT_FOLDER'checkpoints/model.ckpt-'
  CHECKPOINT_FILE+=$i
  python evaluate.py \
    --data_dir=${DATASET_DIR} \
    --data_list=${DATA_LIST} \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --evaluate_log_file=${EVALUATE_LOG_FILE} \
    --gpu=${GPU}
done
