#!/bin/bash

DATASET_DIR=~/SS/Datasets/cityscapes-images/
DATA_LIST=~/SS/mobilenet_psp/list/eval_list.txt
CHECKPOINT_FILE=checkpoints/model.ckpt-43
EVALUATE_LOG_FILE=evaluate.log
GPU=1



python evaluate.py \
  --data_dir=${DATASET_DIR}
  --data_list=${DATA_LIST}
  --checkpoint_path=${CHECKPOINT_FILE}
  --evaluate_log_file=${EVALUATE_LOG_FILE}
  --gpu=${GPU}
