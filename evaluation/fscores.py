## all to npz file and calculate scores due to the memory issue
import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.sparse import load_npz, save_npz, csr_matrix
from pecos.utils import smat_util
#turn off warnings since division error (divided by 0) always heppens
import warnings
warnings.filterwarnings("ignore")

def row_topk_csr(data, indices, indptr, ks):
    m = indptr.shape[0] - 1 #number of examples
    max_indices = []
    for i in range(m): #,desc='convert pecos score to |y| best labels',total=m):
        top_inds = np.argsort(data[indptr[i] : indptr[i + 1]])[::-1][:int(ks[i])]
        max_indices.append(indices[indptr[i] : indptr[i + 1]][top_inds])

    return max_indices

def row_fixk_csr(data, indices, indptr, k):
    m = indptr.shape[0] - 1 #number of examples
    max_indices = []
    #for i in tqdm(range(m),desc='convert pecos score to |y| best labels',total=m):
    for i in range(m):
        top_inds = np.argsort(data[indptr[i] : indptr[i + 1]])[::-1][:k]
        max_indices.append(indices[indptr[i] : indptr[i + 1]][top_inds])

    return max_indices


def row_thold_csr(data, indices, indptr, thold):
    m = indptr.shape[0] -1
    max_indices = []
    for i in range(m):#, desc='get thold idxs for pecos', total=m):
        top_inds = data[indptr[i]:indptr[i+1]] > thold
        max_indices.append(indices[indptr[i]:indptr[i+1]][top_inds])
    return max_indices

def cls_to_npz(path):
    npz_path = os.path.join(path,"CL_P.npz")
    npy_path = os.path.join(path,"CL_P.npy")
    #save as npz
    pred_arrays = np.load(npy_path)
    pred_arrays = pred_arrays > 0.5
    pred_arrays = csr_matrix(np.array(pred_arrays))
    save_npz(npz_path,pred_arrays)


def read_label_map(path):
    labels = []
    with open(path,'r') as f:
        for line in f:
            labels+=line.replace("\n","").split(" ")
    return np.array(labels)

def read_cluster_map(path):
    clusters = []
    with open(path,'r') as f:
        for line in f:
            clusters.append(line.replace("\n",""))

    #CLUSTER SORTING IS BASED ON FREQUENCY IN LABEL MAP
    clusters = sorted(clusters, key = clusters.count, reverse=True)
    clusters = np.array(list(dict.fromkeys(clusters)))

    return clusters

def get_true_cluster_npz(cluster_path):
    #Set path and file names
    cls_file_name = 'cluster_test_kmeans'
    cls_map_name = 'label_map_cluster_kmeans.txt'

    cls_map_path =os.path.join(cluster_path,cls_map_name)
    cls_npz_path = os.path.join(cluster_path,cls_file_name+".npz")
    cls_path = os.path.join(cluster_path,cls_file_name)

    maps = read_cluster_map(cls_map_path)
    maps = {item: idx for idx, item in enumerate(maps)}
    if not os.path.exists(cls_npz_path):
        cls_arrays = []
        with open(os.path.join(cls_path),'r') as f:
            for line in f:
                cls = list(set(line.replace("\n","").split(" ")))
                cls_arr = np.zeros(len(maps))
                #update 1 where preds in maps pred_arr
                idxs=[maps.get(item) for item in cls]
                cls_arr[idxs] = 1
                cls_arrays.append(cls_arr)
        #save as npz
        cls_arrays = csr_matrix(np.array(cls_arrays))
        save_npz(cls_npz_path,cls_arrays)

    return cls_npz_path, cls_map_path

def t5_to_npz(map_path,pred_path,true_path,type='label'):
    #Load label map
    if type=='label':
        maps = read_label_map(map_path)
        npz_name="P.npz"
    elif type=='cluster':
        maps = read_cluster_map(map_path)
        #for txt we use cluster with "<>" so we manually add it
        maps = np.array(["<"+x+">" for x in maps])
        npz_name="CL_P.npz"

    maps = {item: idx for idx, item in enumerate(maps)}
    pred_arrays = []
    pred_scores = []

    true_npz = load_npz(true_path)
    zero_csr_matrix = csr_matrix(np.zeros(true_npz.shape[1]))
    pred_csr_npz = true_npz.multiply(zero_csr_matrix).tolil()
    pred_score_csr_npz = true_npz.multiply(zero_csr_matrix).tolil()
    with open(os.path.join(pred_path,'eval_preds_test.txt'),'r') as f:
        for idx, line in tqdm(enumerate(f), desc='load eval file to convert npz'):
            line = line.replace("</s>","").replace("<pad>","").replace("<"," <").replace(">","> ").replace("  "," ").replace("\n","").strip().split()
            preds = list(dict.fromkeys([x for x in line if x in maps]))

            idxs=[maps.get(item) for item in preds]
            pred_csr_npz[idx,idxs] = 1
            pred_score_csr_npz[idx,idxs] = [-(i+1) for i in range(len(preds))]

    #save as npz
    pred_csr_npz = csr_matrix(pred_csr_npz)
    pred_score_csr_npz = csr_matrix(pred_score_csr_npz)
    save_npz(os.path.join(pred_path,npz_name),pred_csr_npz)
    save_npz(os.path.join(pred_path,'score.npz'),pred_score_csr_npz)

def axml_to_npz(map_path,pred_path,true_path,thold=0):
    #For AttentionXML
    maps = read_label_map(map_path)
    maps = {item: idx for idx, item in enumerate(maps)}

    pred_txt_path=os.path.join(pred_path,"L.npy")
    pred_scr_path=os.path.join(pred_path,'P.npy')
    pred_txt_arrays = np.load(pred_txt_path, allow_pickle=True)
    pred_scr_arrays = np.load(pred_scr_path, allow_pickle=True)
    true_npz = load_npz(true_path)
    true_len = true_npz.sum(axis=1).A1 #matrix to array

    zero_csr_matrix = csr_matrix(np.zeros(true_npz.shape[1]))
    pred_csr_npz = true_npz.multiply(zero_csr_matrix).tolil()
    pred_score_csr_npz = true_npz.multiply(zero_csr_matrix).tolil()

    for idx, (pred,score) in tqdm(enumerate(zip(pred_txt_arrays,pred_scr_arrays)),total=len(pred_scr_arrays),desc='convert npy to npz for axml'):
        pred_arr = np.zeros(len(maps))
        pred_scr = np.zeros(len(maps))

        #pred_len = sum(score>0.5)
        if thold > 0:
            pred_len = int(sum(score > thold))
        else:
            pred_len = int(true_len[idx])

        idxs=[maps.get(item) for item in pred[:pred_len]]
        try:
            pred_csr_npz[idx,idxs] = 1
            pred_score_csr_npz[idx,idxs] = score[:pred_len]
        except TypeError:
            continue

    pred_csr_npz = csr_matrix(pred_csr_npz)
    pred_score_csr_npz = csr_matrix(pred_score_csr_npz)
    save_npz(os.path.join(pred_path,"P.npz"),pred_csr_npz)
    #save scores for @10 scores
    save_npz(os.path.join(pred_path,"score.npz"),pred_score_csr_npz)

def pecos_to_npz(map_path,pred_path,true_path,thold=0):
    #For other pecos baslines
    maps = read_label_map(map_path)
    pred_csr_path=os.path.join(pred_path,'P.npz')
    pred_csr_npz = load_npz(pred_csr_path)
    if thold > 0:
        idxs = row_thold_csr(pred_csr_npz.data, pred_csr_npz.indices, pred_csr_npz.indptr, thold)
    else:
        true_npz = load_npz(true_path)
        true_len = true_npz.sum(axis=1).A1 #matrix to array
        idxs = row_topk_csr(pred_csr_npz.data, pred_csr_npz.indices, pred_csr_npz.indptr, true_len)
    zero_csr_matrix = csr_matrix(np.zeros(pred_csr_npz.shape[1]))
    pred_csr_npz = pred_csr_npz.multiply(zero_csr_matrix)
    pred_csr_npz = pred_csr_npz.tolil()
    for i,idx in tqdm(enumerate(idxs),total=len(idxs),desc='change to 1-0 npz'):
        try:
            pred_csr_npz[i,idx] = 1
        except TypeError:
            continue
    pred_csr_npz = csr_matrix(pred_csr_npz)
    return pred_csr_npz

def gpt_to_npz(map_path,pred_path,true_path,type='label'):
    #Load label map
    maps = read_label_map(map_path)
    npz_name="P.npz"

    maps = {item: idx for idx, item in enumerate(maps)}
    pred_arrays = []
    pred_scores = []

    true_npz = load_npz(true_path)
    zero_csr_matrix = csr_matrix(np.zeros(true_npz.shape[1]))
    pred_csr_npz = true_npz.multiply(zero_csr_matrix).tolil()
    pred_score_csr_npz = true_npz.multiply(zero_csr_matrix).tolil()
    with open(os.path.join(pred_path),'r') as f:
        for idx, line in tqdm(enumerate(f), desc='load eval file to convert npz'):
            line = line.replace("</s>","").replace("<pad>","").replace("<"," <").replace(">","> ").replace("  "," ").replace("\n","").strip().split()
            preds = list(dict.fromkeys([x for x in line if x in maps]))

            idxs=[maps.get(item) for item in preds]
            pred_csr_npz[idx,idxs] = 1
            pred_score_csr_npz[idx,idxs] = [-(i+1) for i in range(len(preds))]

    #save as npz
    pred_csr_npz = csr_matrix(pred_csr_npz)
    pred_score_csr_npz = csr_matrix(pred_score_csr_npz)
    pred_path_ =  '/'.join(pred_path.split("/")[:-1])

    save_npz(os.path.join(pred_path_,npz_name),pred_csr_npz)
    save_npz(os.path.join(pred_path_,'score.npz'),pred_score_csr_npz)


def get_scores(pred_path,true_path,map_path,type='label',thold=0,save_labelwise=False):
    #Save npz file if not exists
    if type=='label':
        npz_name = "P.npz"
        npy_name = "P.npy"

    elif type=='cluster':
        npz_name = "CL_P.npz"
        npy_name = "CL_P.npy"

    if 'gpt' in pred_path:
        pred_path_ =  '/'.join(pred_path.split("/")[:-1])
        pred_path_npy = os.path.join(pred_path_,npy_name)
        pred_path_npz = os.path.join(pred_path_,npz_name)

    else:
        pred_path_npy = os.path.join(pred_path,npy_name)
        pred_path_npz = os.path.join(pred_path,npz_name)

    #convert bce cluster .npy to .npz for the efficiency
    if os.path.exists(pred_path_npy) and 't5' in pred_path_npy:
        cls_to_npz(pred_path)

    #Format to npz and score.npz for @10 metrics
    if 'axml' in pred_path:
        axml_to_npz(map_path,pred_path,true_path,thold)
        pred_npz = load_npz(pred_path_npz)

    elif 'gpt' in pred_path:
        gpt_to_npz(map_path,pred_path,true_path,type)
        pred_npz = load_npz(pred_path_npz)

    elif 't5' in pred_path or 'bart' in pred_path:
        if 'bce' not in pred_path or type=='label':
            t5_to_npz(map_path,pred_path,true_path,type)
        pred_npz = load_npz(pred_path_npz)

    else:
        pred_npz = pecos_to_npz(map_path,pred_path,true_path,thold)


    true_npz = load_npz(true_path)
    muls = true_npz.multiply(pred_npz)

    #Get micro scores first
    micro_prec = muls.sum() / pred_npz.sum()
    micro_recl = muls.sum() / true_npz.sum()
    micro_f1 = (2* (micro_prec*micro_recl)) / (micro_prec+micro_recl)
    #Get macro scores
    per_prec = np.nan_to_num(muls.sum(axis=0) / pred_npz.sum(axis=0),0)
    per_recl = np.nan_to_num(muls.sum(axis=0) / true_npz.sum(axis=0),0)
    non_zero_labels = np.nonzero(true_npz.sum(axis=0).A1)

    per_prec = per_prec.A1
    per_recl = per_recl.A1
    per_f1 = np.nan_to_num([(2*x*y)/(x+y) for x,y in zip(per_prec,per_recl)])

    #macro_prec = np.mean(per_prec)
    #macro_recl = np.mean(per_recl)
    #macro_f1 = np.mean(per_f1)

    macro_prec= np.mean(per_prec[non_zero_labels])
    macro_recl= np.mean(per_recl[non_zero_labels])
    macro_f1 = np.mean(per_f1[non_zero_labels])
    #macro_f1 = (2* (macro_prec*macro_recl)) / (macro_prec+macro_recl)

    #import pdb;pdb.set_trace();
    print("%s Micro prec: %.3f, rec: %.3f, f1: %.3f" %(type, micro_prec,micro_recl,micro_f1))
    print("%s Macro prec: %.3f, rec: %.3f, f1: %.3f" %(type, macro_prec,macro_recl,macro_f1))

    #Save labelwise precision / recall scores in pred_path
    if save_labelwise:
        np.save(os.path.join(pred_path,'per_prec.npy'),per_prec)
        np.save(os.path.join(pred_path,'per_recl.npy'),per_recl)

    #get @10 scores
    if type=='label' and 'gpt' not in pred_path:
        if 't5' in pred_path or 'axml' in pred_path or 'bart' in pred_path:
            file_nm = 'score.npz'
        else:
            file_nm = 'P.npz'

        pred_csr_npz = load_npz(os.path.join(pred_path,file_nm))
#        metric = smat_util.Metrics.generate(true_npz, pred_csr_npz, topk=10)
#        for idx in [0,2,4,9]:
#            print("prec@%s: "%(idx+1), round(metric.prec[idx],3),
#                  " rec@%s: "%(idx+1), round(metric.recall[idx],3),
#                 " f1@%s: "%(idx+1), round(2*(metric.prec[idx]*metric.recall[idx])/(metric.prec[idx]+metric.recall[idx]),3)
#                  )

        #test
        print("Handwritten @k")
        for k in [1,3,5,10]:
            k_idxs = row_fixk_csr(pred_csr_npz.data, pred_csr_npz.indices, pred_csr_npz.indptr, k)
            zero_csr_matrix = csr_matrix(np.zeros(pred_csr_npz.shape[1]))
            pred_csr_npz_ = pred_csr_npz.multiply(zero_csr_matrix).tolil()
            for i,idx in enumerate(k_idxs):
                try:
                    pred_csr_npz_[i,idx] = 1
                except TypeError:
                    continue

            pred_scr_npz_ = csr_matrix(pred_csr_npz_)
            nom = true_npz.multiply(pred_csr_npz_).sum()
            prec_denom = pred_csr_npz_.sum()
            rec_denom = true_npz.sum()

            mi_prec = nom / prec_denom
            mi_rec = nom / rec_denom

            print("prec@%s: "%(k), round(mi_prec,3),
                  "rec@%s: "%(k), round(mi_rec,3),
                  "f1@%s: "%(k), round(2*(mi_prec*mi_rec)/(mi_prec+mi_rec),3)
                  )


        print("Few-Shot scores")
        trn_true_npz = load_npz(true_path.replace("tst",'trn'))
        trn_cnts = trn_true_npz.sum(axis=0).A1
        tst_cnts = true_npz.sum(axis=0).A1
        for shot in [0,1,3,5,10]:
            if shot == 0:
                idxs = [idx for idx, x in enumerate(trn_cnts) if x == 0 ]
                idxs_tst = [idx for idx, (x,y) in enumerate(zip(trn_cnts,tst_cnts)) if x == 0 and y > 0]

            else:
                idxs = [idx for idx, x in enumerate(trn_cnts) if x < shot+1 and x > 0]
                idxs_tst = [idx for idx, (x,y) in enumerate(zip(trn_cnts,tst_cnts)) if x<shot+1 and x >0 and y > 0]


            pred_npz_ = pred_npz[:,idxs]
            true_npz_ = true_npz[:,idxs]
            muls_ = true_npz_.multiply(pred_npz_)
            #Get micro scores first
            micro_prec = np.nan_to_num(muls_.sum() / pred_npz_.sum())
            micro_recl = np.nan_to_num(muls_.sum() / true_npz_.sum())

            micro_f1 = np.nan_to_num((2* (micro_prec*micro_recl)) / (micro_prec+micro_recl))

            #Get macro scores
            per_prec = np.nan_to_num(muls_.sum(axis=0) / pred_npz_.sum(axis=0),0)
            per_recl = np.nan_to_num(muls_.sum(axis=0) / true_npz_.sum(axis=0),0)

            macro_prec = np.nan_to_num(np.mean(per_prec))
            macro_recl = np.nan_to_num(np.mean(per_recl))

            macro_f1 = np.nan_to_num((2* (macro_prec*macro_recl)) / (macro_prec+macro_recl))
            print("Few-Shot %d (%d out of %d total labels, %d in test set)" %(shot,len(idxs),len(tst_cnts),len(idxs_tst)))
            print("%s Micro prec: %.3f, rec: %.3f, f1: %.3f" %(type, micro_prec,micro_recl,micro_f1))
            print("%s Macro prec: %.3f, rec: %.3f, f1: %.3f" %(type, macro_prec,macro_recl,macro_f1))


def main(pred_path,true_path,map_path,model,label_type,cluster_path=None,cluster_map_path=None,thold=0,save_labelwise=False):

    #label always
    get_scores(pred_path,true_path,map_path,type='label',thold=thold, \
                save_labelwise=save_labelwise)

    #cluster if exists
    if label_type !='base':
        get_scores(pred_path,cluster_path,cluster_map_path,type='cluster',thold=thold)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default=None, required=True, help="Directory of Pred file")
    parser.add_argument("--true_path", type=str, default=None, required=True, help="Directory of True file")
    parser.add_argument("--map_path", type=str, default=None, required=True, help="Directory of label map")
    parser.add_argument("--cluster_path", type=str, default=None, help="Directory of cluster True file")
    parser.add_argument("--model", type=str, default=None, required=True, help="Name of model")
    parser.add_argument("--label_type", type=str,default=None, help="Label type for t5 models")
    parser.add_argument("--cluster", type=str, default=None)
    parser.add_argument("--cluster_feat", type=str, default=None)
    parser.add_argument("--cluster_size", type=str, default=None)
    parser.add_argument("--thold", type=float, default=0)
    parser.add_argument("--save_labelwise", action='store_true')
    args = parser.parse_args()


    #generate true path separately
    cluster_path=None
    cluster_map_path=None #default it is None
    if 'base' not in args.label_type:
        cluster_path,cluster_map_path = get_true_cluster_npz(args.cluster_path)

    main(args.pred_path,args.true_path,args.map_path, \
            args.model,args.label_type,cluster_path,cluster_map_path,args.thold, \
            args.save_labelwise)



