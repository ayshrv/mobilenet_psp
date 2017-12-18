#!/bin/bash

# python ~/SS/mobilenet_psp/evaluate.py --gpu=1 --checkpoint_path=checkpoints/model.ckpt-43

DATASET_DIR=~/SS/Datasets/cityscapes-images/
DATA_LIST=list/eval_list.txt
CHECKPOINT_FOLDER=logs/train1-Fine-Full-Momentum/
EPOCHS='43'
GPU=1

CHECKPOINT_FILE=$CHECKPOINT_FOLDER'model.ckpt-'$EPOCHS
EVALUATE_LOG_FILE=$CHECKPOINT_FOLDER'evaluate.log'


python evaluate.py \
  --data_dir=${DATASET_DIR} \
  --data_list=${DATA_LIST} \
  --checkpoint_path=${CHECKPOINT_FILE} \
  --evaluate_log_file=${EVALUATE_LOG_FILE} \
  --gpu=${GPU}
