This repository contains codes of Jung et al's EACL 2023 paper titled "Cluster-Guided Label Generation in Extreme Multi-Label Classification"

## Requirements
We use python 3.6.5. Please run ```pip install -r requirement.txt``` to install python dependencies.


## Running XLGen with EUR-Lex dataset
We provide an example of XLGen training/evaluation on EUR-Lex dataset with t5-base model.
To test with other benchmark datasets (e.g., Wiki10-31K, AmazonCat-13K, Wiki-500K) or t5 model (e.g., t5-large), simply change the corresponding arguments.

1. Data Download
```
bash ./download_data/download_data.sh EUR-Lex
```

2. Get Label Clusters

2.1 Get t5 label representations
```
bash ./cluster/run_t5_rep.sh EUR-Lex t5-base
```

2.2 Get kmeans label clusters
```
bash ./cluster/run_cluster.sh EUR-Lex t5-base
```

3. Running XLGen models

3.1 Train XLGen_ base / XLGen_bcl / XGLEN_mcg models
```
bash ./xlgen/run_train.sh EUR-Lex t5-base base
bash ./xlgen/run_train.sh EUR-Lex t5-base bcl
bash ./xlgen/run_train.sh EUR-Lex t5-base mcg
```

3.2 Inference for XLGen_ base / XLGen_bcl / XGLEN_mcg models
```
bash ./xlgen/run_test.sh EUR-Lex t5-base base
bash ./xlgen/run_test.sh EUR-Lex t5-base bcl
bash ./xlgen/run_test.sh EUR-Lex t5-base mcg
```

4. Evaluation with Fscores (+ precision@K)
```
bash ./evaluation/run_fscores.sh EUR-Lex t5-base base
bash ./evaluation/run_fscores.sh EUR-Lex t5-base bcl
bash ./evaluation/run_fscores.sh EUR-Lex t5-base mcg
```

## Citation
    @article{jung2023cluster,
    title={Cluster-Guided Label Generation in Extreme Multi-Label Classification},
    author={Jung, Taehee and Kim, Joo-Kyung and Lee, Sungjin and Kang, Dongyeop},
    journal={arXiv preprint arXiv:2302.09150},
    year={2023}
    }
