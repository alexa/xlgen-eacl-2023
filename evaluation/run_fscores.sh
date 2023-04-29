#!/bin/bash
target=freq_forward
data=$1
model=$2
label=$3
cluster_feat=$model #Fix it for experiments;


pred_path=$HOME/data/xml/output/$data/$model/$target/$label/
cluster_path=$HOME/data/xml/data/$data/cluster
true_path=$HOME/data/xml/data/$data/pecos/Y.tst.npz
map_path=$HOME/data/xml/data/$data/label_map.txt

echo $pred_path
python ./fscores.py \
        --pred_path $pred_path \
        --true_path $true_path \
        --map_path $map_path \
        --cluster_path $cluster_path \
        --model $model \
        --label_type $label \
        --save_labelwise

