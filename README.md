
data list : EUR-Lex, Wiki10-31K, AmazonCat-13K, Wiki-500K
label type list : org, bce-sep, decoder-sep

How to run the code?

1. download data
./download/download_data.sh ${DATA_NAME}
e.g., ./download/download_data.sh EUR-Lex

2. make t5 representation for cluster
./cluster/run_t5_rep.sh ${DATA_NAME} ${MODEL_NAME}
e.g., ./cluster/run_t5_rep.sh EUR-Lex t5-large

3. train and predict cluster with t5 representation
./cluster/run_cluster.sh ${DATA_NAME} ${MODEL_NAME}
e.g., ./cluster/run_cluster.sh EUR-Lex t5-large kmeans

4. train t5 model
./run_train.sh ${DATA_NAME} ${MODEL_NAME} ${LABEL_TYPE} ${CLUSTER_NAME} ${GPU_NUMS} ${EPOCH_SIZE}
e.g., ./t5xml/run_train.sh EUR-Lex t5-large bce-sep kmeans 0,1,2,3 5

5. predict with t5 model : WE SHOULD RUN THIS SEPARATELY FROM ./run_train.sh as we add epoch size as a param
./run_test.sh ${DATA_NAME} ${MODEL_NAME} ${LABEL_TYPE} ${CLUSTER_NAME} ${GPU_NUMS} ${EPOCH_SIZE}
e.g., ./t5xml/run_test.sh EUR_Lex t5-large bce-sep kmeans 0,1,2,3 3
--> this will generate outputs with model trained epoch=3

6. evaluation
./run_fscores.sh ${DATA_NAME} ${MODEL_NAME} ${LABEL_TYPE} ${CLUSTER_NAME} ${EPOCH_SIZE}
e.g., ./run_Fscores_t5.sh EUR-Lex t5-large bce-sep kmeans 3
