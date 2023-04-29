#!/bin/bash

TARGET_ORDER="freq_forward" #"freq_backward" "shuffle"
DATA_DIR=$HOME/data/xml/data/
MODEL_DIR=$HOME/data/xml/model/
OUTPUT_DIR=$HOME/data/xml/output/

TASK=$1 #"Eur-Lex" "Wiki10-31K" "AmazonCat-13K" "Wiki-500K"
MODEL=$2 #"t5-base" "t5-large"
LABEL=$3

#Hard coding MAXOUTLEN based on dataset
if [ $TASK == "EUR-Lex" ]
then
    MAXOUTLEN=90
else
    MAXOUTLEN=165
fi

echo "Running ... ${TASK} ${MODEL} ${LABEL} ${TARGET_ORDER}"
CUDA_VISIBLE_DEVICES=0 \
    python ./main.py \
    --model_name_or_path ${MODEL} \
    --model_name ${MODEL} \
    --task_name ${TASK} \
    --do_train \
    --do_lower_case \
    --data_dir ${DATA_DIR}/${TASK} \
    --max_out_seq_len $MAXOUTLEN \
    --logging_steps 10000 \
    --save_steps 10000 \
    --learning_rate 2e-4 \
    --overwrite_output_dir \
    --overwrite_cache \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --output_dir ${MODEL_DIR}/${TASK}/${MODEL}/${TARGET_ORDER}/${LABEL} \
    --eval_output_dir ${OUTPUT_DIR}/${TASK}/${MODEL}/${TARGET_ORDER}/${LABEL} \
    --label_type ${LABEL} \
    --decode_order ${TARGET_ORDER} \
    --num_beams 5 \
