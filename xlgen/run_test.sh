#!/bin/bash
TARGET_ORDER="freq_forward" #"freq_backward" "shuffle"
DATA_DIR=$HOME/data/xml/data
MODEL_DIR=$HOME/data/xml/model
OUTPUT_DIR=$HOME/data/xml/output

TASK=$1 # "Wiki10-31K" "EUR-Lex"
MODEL=$2
LABEL=$3

echo "Testing ... ${TASK} ${MODEL} ${LABEL} ${TARGET_ORDER}"

CUDA_VISIBLE_DEVICES=0 \
    python ./main.py \
    --model_name_or_path ${MODEL_DIR}/${TASK}/${MODEL}/${TARGET_ORDER}/${LABEL} \
    --model_name ${MODEL} \
    --task_name ${TASK} \
    --do_eval \
    --overwrite_cache \
    --do_lower_case \
    --data_dir ${DATA_DIR}/${TASK} \
    --output_dir ${MODEL_DIR}/${TASK}/${MODEL}/${TARGET_ORDER}/${LABEL} \
    --eval_output_dir ${OUTPUT_DIR}/${TASK}/${MODEL}/${TARGET_ORDER}/${LABEL} \
    --label_type ${LABEL} \
    --decode_order ${TARGET_ORDER} \
    --num_beams 5


