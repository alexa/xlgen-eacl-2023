#!/bin/bash

TARGET_ORDER="freq_forward"
DATA_DIR=$HOME/data/xml/data/
MODEL_DIR=$HOME/data/xml/model/
OUTPUT_DIR=$HOME/data/xml/output/

TASK=$1
MODEL=$2
BATCH=128
MAXLEN=500
MAXOUTLEN=200
LABEL="base"

echo "GET T5 REPRESENTATION FOR CLUSTERING FROM ... ${TASK} ${MODEL}"
CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
python ../xlgen/main.py \
    --model_name_or_path ${MODEL} \
    --model_name ${MODEL} \
    --task_name ${TASK} \
    --do_lower_case \
    --data_dir ${DATA_DIR}/${TASK} \
    --cache_data_dir ${DATA_DIR}/${TASK}/cache/${TARGET_ORDER}/${LABEL} \
    --max_seq_len $MAXLEN \
    --max_out_seq_len $MAXOUTLEN \
    --per_device_train_batch_size=$BATCH   \
    --label_type ${LABEL} \
    --decode_order ${TARGET_ORDER} \
    --get_rep \
    --output_dir ${MODEL_DIR} \
    --overwrite_output_dir

