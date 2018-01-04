#!/bin/bash

#Method 2
DATASET_DIR=~/SS/Datasets/cityscapes-images/
DATA_LIST=list/eval_list.txt
CHECKPOINT_FOLDER=logs/train1-Fine-Full-Momentum/ #The folder which contains the checkpoints
EVALUATE_LOG_FILE=$CHECKPOINT_FOLDER'evaluate.log'
GPU=1
EPOCHS=( 7 10 34)  #list of particular epochs to evaluate. In this case 7, 10 and 34

for i in "${EPOCHS[@]}"
do
  CHECKPOINT_FILE=$CHECKPOINT_FOLDER'checkpoints/model.ckpt-'
  CHECKPOINT_FILE+=$i
  python evaluate.py \
    --data_dir=${DATASET_DIR} \
    --data_list=${DATA_LIST} \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --evaluate_log_file=${EVALUATE_LOG_FILE} \
    --gpu=${GPU}
done
