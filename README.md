This repository contains codes of Jung et al's EACL 2023 paper titled "Cluster-Guided Label Generation in Extreme Multi-Label Classification"

## Requirements
We use python 3.6.5. Please run ```pip install -r requirement.txt``` to install python dependencies.


## Running XLGen with EUR-Lex dataset
We provide an example of XLGen training/evaluation on EUR-Lex dataset with t5-base model.
To test with other benchmark datasets (e.g., Wiki10-31K, AmazonCat-13K, Wiki-500K) or t5 model (e.g., t5-large), simply change the corresponding arguments.

#### Data Download
```
bash ./download_data/download_data.sh EUR-Lex
```

#### Get Label Clusters

1. Get t5 label representations
```
bash ./cluster/run_t5_rep.sh EUR-Lex t5-base
```

2. Get kmeans label clusters
```
bash ./cluster/run_cluster.sh EUR-Lex t5-base
```

#### Running XLGen models

1. Train XLGen-base / XLGen-bcl / XGLEN-mcg
```
bash ./xlgen/run_train.sh EUR-Lex t5-base base
bash ./xlgen/run_train.sh EUR-Lex t5-base bcl
bash ./xlgen/run_train.sh EUR-Lex t5-base mcg
```

2. Inference for XLGen-base / XLGen-bcl / XGLEN-mcg
```
bash ./xlgen/run_test.sh EUR-Lex t5-base base
bash ./xlgen/run_test.sh EUR-Lex t5-base bcl
bash ./xlgen/run_test.sh EUR-Lex t5-base mcg
```

#### Evaluation
```
bash ./evaluation/run_fscores.sh EUR-Lex t5-base base
bash ./evaluation/run_fscores.sh EUR-Lex t5-base bcl
bash ./evaluation/run_fscores.sh EUR-Lex t5-base mcg
```

## Citation
   @Inproceedings{Jung2023,
    author = {Taehee Jung and Joo-Kyung Kim and Sungjin Lee and Dongyeop Kang},
    title = {Cluster-guided label generation in extreme multi-label classification},
    year = {2023},
    url = {https://www.amazon.science/publications/cluster-guided-label-generation-in-extreme-multi-label-classification},
    booktitle = {EACL 2023},
    }
