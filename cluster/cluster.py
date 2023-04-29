### Create clusters (kmeans vs hieararchical clustering) using datasets
### We borrow some classes from pecos for preprocessing data
from pecos.xmc.xlinear.model import XLinearModel
from pecos.xmc import LabelEmbeddingFactory
import numpy as np
import argparse
import os
from sklearn.cluster import KMeans, AgglomerativeClustering

def load_label(label_path):
    label_list = list()
    with open(label_path,'r') as f:
        for idx, line in enumerate(f):
            label_list.append(line.replace("\n",""))
    return label_list

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_trn", type=str, required=True, help="Path to x train matrix")
    parser.add_argument("--y_trn", type=str, required=True, help="Path to y train matrix")
    parser.add_argument("--cluster_type", type=str, default='kmeans',
            help="type of clustering method")
    parser.add_argument("--method", type=str, default='pifa',
            help="type for cluster input, default is pifa")
    parser.add_argument("--data_path", type=str, required=True,
            help="Path to data")
    parser.add_argument("--train_data_path", type=str, default=None,
            help="Path to train data if it exists seperately")
    parser.add_argument("--save_path", type=str, required=True,
            help="Path to label info")
    parser.add_argument("--data", type=str, required=True, help="Data name")
    parser.add_argument("--num_cls", type=int, default=8, help='Number of clusters')
    args = parser.parse_args()

    #Use Pecos dataset
    #Load x, y dataset
    X = XLinearModel.load_feature_matrix(args.x_trn)
    Y = XLinearModel.load_label_matrix(args.y_trn)
    if X.dtype!='float32':
        X=X.astype("float32")

    #Construct label features for clustering
    label_feat = LabelEmbeddingFactory.create(Y,X, method=args.method)
    label_list = load_label(os.path.join(args.data_path,'label_map.txt'))
    n_clusters = args.num_cls

    if label_feat.dtype!='float32':
        label_feat=label_feat.astype("float32")
    #n_clusters = int(label_feat.shape[0]/args.num_labs_in_cls)
    if 'hcluster' in args.cluster_type:
        cluster = AgglomerativeClustering(n_clusters=n_clusters, \
                                            affinity='euclidean',\
                                            linkage='ward')
    elif 'kmeans' in args.cluster_type:
        cluster = KMeans(n_clusters=n_clusters)
    if 'npz' in args.x_trn:
        pred_cluster = cluster.fit_predict(label_feat.toarray())
    elif 'npy' in args.x_trn:
        pred_cluster = cluster.fit_predict(label_feat)

    #out cluster map -->
    cluster_map = args.cluster_type
    with open(os.path.join(args.save_path,"label_map_cluster_%s.txt" \
                            %(cluster_map)),'w') as f:
        for clust in pred_cluster:
            clust = str(clust)
            f.write('c' + clust + '\n')

    #get cluster info for train/dev/test set
    #get label dict --> map to cluster
    label_dict = dict()
    for (lab_,cls_) in zip(label_list,pred_cluster):
        label_dict[lab_] = 'c' + str(cls_)

    for set_type in ['train','test']:
        cls_out = []
        if set_type=='train' and args.train_data_path is not None:
            data_path = args.train_data_path
        else:
            data_path = args.data_path

        if not os.path.exists(os.path.join(args.save_path,"cluster_%s_%s" \
                                            %(set_type,cluster_map))):
            with open(os.path.join(data_path,"%s_labels.txt"%(set_type)),'r') as f:
                for line in f:
                    labels=line.split(' ')
                    cls_ = ''
                    for lab_ in labels:
                        cls_+=' ' + label_dict[lab_.replace("\n","")]
                    cls_out.append(cls_.strip())

            with open(os.path.join(args.save_path,"cluster_%s_%s"
                                %(set_type,cluster_map)),'w') as f:
                for clus in cls_out:
                    f.write(clus+'\n')

