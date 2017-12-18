#!/bin/bash

# python ~/SS/mobilenet_psp/evaluate.py --gpu=1 --checkpoint_path=checkpoints/model.ckpt-43

# Method 1
# DATASET_DIR=~/SS/Datasets/cityscapes-images/
# DATA_LIST=list/eval_list.txt
# CHECKPOINT_FOLDER=logs/train1-Fine-Full-Momentum/
# EPOCHS=43
# GPU=1
#
# EVALUATE_LOG_FILE=$CHECKPOINT_FOLDER'evaluate.log'
#
# CHECKPOINT_FILE=$CHECKPOINT_FOLDER'checkpoints/model.ckpt-'
# CHECKPOINT_FILE+=$EPOCHS
#
# python evaluate.py \
#   --data_dir=${DATASET_DIR} \
#   --data_list=${DATA_LIST} \
#   --checkpoint_path=${CHECKPOINT_FILE} \
#   --evaluate_log_file=${EVALUATE_LOG_FILE} \
#   --gpu=${GPU}


#Method 2
DATASET_DIR=~/SS/Datasets/cityscapes-images/
DATA_LIST=list/eval_list.txt
CHECKPOINT_FOLDER=logs/train1-Fine-Full-Momentum/
EVALUATE_LOG_FILE=$CHECKPOINT_FOLDER'evaluate.log'
GPU=1


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
