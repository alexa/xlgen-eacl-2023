data=$1 #Wiki10-31K AmaconCat-13K Wiki-500K Amazon-670K Amazon-3M
model=$2 #t5-base t5-large
data_path=$HOME/data/xml/data/${data} #/pecos
Y_path=${data_path}/pecos/Y.trn.npz
save_path=${data_path}/cluster
cluster_size=30

mkdir -p ${save_path}

echo "generate kmeans cluster for $data cluster size $cluster_size"

X_path=${data_path}/X.trn.${model}.npy

python ./cluster.py \
    --x_trn ${X_path} \
    --y_trn ${Y_path} \
    --data_path ${data_path} \
    --data ${data} \
    --num_cls ${cluster_size}\
    --save_path ${save_path}
