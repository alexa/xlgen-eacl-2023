import csv
import logging
import os
import sys
import argparse
import glob
import pprint
import pdb
import numpy as np
import os
from collections import defaultdict
import random
import math
import coloredlogs, logging
import torch
from copy import copy, deepcopy

from colorama import Fore,Style
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

import torch
from torch.utils.data import TensorDataset
from typing import Iterable
from IPython.core.display import display, HTML, Image

from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from captum.attr._utils.visualization import *


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                      datefmt='%m/%d/%Y %H:%M:%S',
                      level=logging.INFO)

def _set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def info(logger, training_args):
    logger.info(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            highlight(training_args.local_rank),
            highlight(training_args.device),
            highlight(training_args.n_gpu),
            highlight(bool(training_args.local_rank != -1)),
            highlight(training_args.fp16))
    logger.info("Training/evaluation parameters %s", highlight(training_args))

def _set_barrier(args, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()


def _set_barrier_v2(args, evaluate=False):
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

def _set_fp16(args):
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


def highlight(input):
    input = str(input)
    return str(Fore.YELLOW+str(input)+Style.RESET_ALL)



def _distance_debugging(args):
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()


def _setup_gpu(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)




class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin




#https://gist.github.com/davidefiocco/3e1a0ed030792230a33c726c61f6b3a5
def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions.detach().cpu().numpy()

def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, label_list):
    # storing couple samples in an array for visualization purposes
    vis = VisualizationDataRecord(
                            attributions,
                            pred,
                            pred_ind,
                            label,
                            label_list[label],
                            attributions.sum(),
                            tokens[:len(attributions)],
                            delta)
    return vis


def visualize_text(datarecords, filename, eval_task) -> None:
    dom = []
    rows = [
        "<table width: 100%>"
        "<tr><th>True Label</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]

    preds_ids,label_ids = [], []
    for datarecord in datarecords:
        color = ''
        if int(datarecord.pred_class) == int(datarecord.true_class):
            color = "bgcolor=#ccccff"
        else:
            color = "bgcolor=#ffb3b3"


        rows.append(
            "".join(
                [
                    "<tr {}>".format(color),
                    format_classname(datarecord.true_class),
                    # format_classname(datarecord.target_class),
                    format_classname(
                        "{0} ({1:.2f})".format(
                            int(datarecord.pred_class), float(datarecord.pred_prob)
                        )
                    ),
                    format_classname(datarecord.attr_class),
                    format_classname("{0:.2f}".format(float(datarecord.attr_score))),
                    format_word_importances(
                        datarecord.raw_input, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )
        preds_ids.append(int(datarecord.pred_class))
        label_ids.append(int(datarecord.true_class))

    result = compute_metrics(eval_task, np.array(preds_ids), np.array(label_ids))
    dom.append("<p>Samples: {}, {}</p>".format(len(preds_ids), result))

    dom.append("".join(rows))
    dom.append("</table>")

    html = HTML("".join(dom))
    with open(filename, 'w') as f:
        f.write(html.data)


def highlight(input):
    return Fore.YELLOW+str(input)+Style.RESET_ALL



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, cluster=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.cluster= cluster


class T5ClusterInputFeatures(object):
    def __init__(
        self,
        input_ids,
        attention_mask,
        target_cluster_labels, #this outputs multiple numbers corresponding labels
        target_input_ids,
        target_attention_mask,
        target_output_ids,
        cluster_id,
        cluster_attention_mask,
        label_id,
        bce_attention_mask, #attention mask for bce training
        input_length
    ):
#         example_index,
#         unique_id,
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.target_cluster_labels = target_cluster_labels
        self.target_input_ids = target_input_ids
        self.target_attention_mask = target_attention_mask
        self.target_output_ids = target_output_ids
        self.cluster_id = cluster_id
        self.cluster_attention_mask = cluster_attention_mask
        self.label_id = label_id
        self.bce_attention_mask = bce_attention_mask
        self.input_length = input_length

class T5ClusterInputTargets(object):
    def __init__(
        self,
        target_input_ids,
        target_attention_mask,
        target_output_ids
    ):
        self.target_input_ids = target_input_ids
        self.target_attention_mask = target_attention_mask
        self.target_output_ids = target_output_ids

"""
https://github.com/huggingface/transformers/issues/3387
input_ids = ['(QUESTION)', Q_W1, Q_W2, ..., '(CONTEXT)', C_W1, C_W2, (EOS), '(PAD)', ...]
attention_masks = [1, 1, 1, ..., 1, 1, 1, 1, ..., 0, ...]
decoder_input_ids = ['(PAD)', W1, W2, ..., '(PAD)', '(PAD)', ...]
decoder_attention_masks = [1, 1, 1, ..., 0, 0, ...]
lm_labels = [W1, W2, ..., '(EOS)', '(PAD)', ...]
EOS token should be added in a context.
And, tokens of the input_ids are [question : Q_W1, Q_W2, ..., context : 'C_W1, C_W2, (EOS), (PAD)', ...]
"""



#ADDED: For shuffled targets
def convert_targets_to_features(targets,
                                tokenizer,
                                max_length_decoder=100,
                                pad_on_left=False,
                                pad_token=0,
                                pad_token_segment_id=0,
                                mask_padding_with_zero=True,
                                verbose=True):

    features = [] #out
    for (ex_index, example) in enumerate(targets):
        target = tokenizer.encode_plus(
                    example, None,
                    add_special_tokens=True,
                    max_length=max_length_decoder-1,
                    truncation=True,
                    return_token_type_ids=False,
                )
        target_ids = target['input_ids']
        target_input_ids = [pad_token] + target_ids
        target_output_ids = target_ids + [tokenizer.eos_token_id]
        target_attention_mask = [1 if mask_padding_with_zero else 0] * len(target_input_ids)

        # Zero-pad up to the sequence length.
        target_padding_length = max_length_decoder - len(target_input_ids)

        if pad_on_left:
            target_input_ids = ([pad_token] * target_padding_length) \
                                + target_input_ids
            target_attention_mask = ([0 if mask_padding_with_zero else 1] \
                    * target_padding_length) + target_attention_mask

            target_output_ids = ([pad_tokens] * padding_length) + target_output_ids

        else:
            target_input_ids = target_input_ids + ([pad_token] * target_padding_length)
            target_attention_mask = target_attention_mask \
                    + ([0 if mask_padding_with_zero else 1] * target_padding_length)

            target_output_ids = target_output_ids + ([pad_token] * target_padding_length)

        assert len(target_input_ids) == max_length_decoder, "Error with input length {} vs {}".format(len(target_input_ids), max_length_decoder)
        assert len(target_attention_mask) == max_length_decoder, "Error with input length {} vs {}".format(
            len(target_attention_mask), max_length_decoder)

        features.append(T5ClusterInputTargets(target_input_ids=target_input_ids,
                        target_attention_mask=target_attention_mask,
                        target_output_ids=target_output_ids)
                )

    return features

## get max cluster size
def get_max_cluster_size(examples):
    cluster_sizes = [len(x.cluster.split(" ")) for x in examples]
    return max(cluster_sizes)

def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_length=512,
                                 max_length_decoder=100,
                                 task=None,
                                 label_list=None,
                                 output_mode=None,
                                 pad_on_left=False,
                                 cluster_list=None,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 model_name=False,
                                 label_type=None,
                                 verbose=True,
                                 max_cluster_size=0,
                                 cluster_noise_ratio=0):
    #label_map = {label : i for i, label in enumerate(label_list)}

    # tokenizer.add_tokens(task)
    # for (ex_index, example) in enumerate(examples):
        # labels = [t for tid, t in enumerate(example.text_a.split()) if t.startswith('<') and t.endswith('>') and tid == 0]
        # for label in labels:
            # tokenizer.add_tokens(label)
    # tokenizer.add_tokens('TEXT')
    # tokenizer.add_tokens('STYLE')
    features = []
    cluster_size = 0
    if cluster_list is not None:
        cluster_size = len(cluster_list)
        cluster_dict = dict()
        for idx, cls_ in enumerate(cluster_list):
            cluster_dict[cls_] = idx


    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text_a_ = example.text_a

        # input format for T5
        # generally simply text or text + clusters
        # for multi type trainings (where cluster_type with 'decoder-xxx'
        # <MultiCluster> text
        # <MultiLabel> text + clusters
        # happy <eos> <pad>.. or noemotion <eos> <pad>..
        style_text_a = ' '.join([t for tid, t in enumerate(text_a_.split()) if t.startswith('<') and t.endswith('>') and tid == 0])[1:-1]
        text_a = ' '.join([t for tid, t in enumerate(text_a_.split()) if (not t.startswith('<') or not t.endswith('>')) and tid > 0])

        if len(style_text_a) > 0:
            text_a = '{} : {}'.format(style_text_a, text_a )
            # prefix_text_a = 'MultiXX: {} '.format(style_text_a)
            # prefix_input_length = len(tokenizer.encode(prefix_text_a))
        else:
            text_a = '{}'.format(text_a)


        inputs = tokenizer.encode_plus(
            text_a, None, add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_token_type_ids=False,
        )

        #Taking cluster first
        if example.cluster is not None:
            cluster_attention_mask = []
            clusters = tokenizer.encode_plus(
                    example.cluster, None,
                    add_special_tokens=True,
                    max_length=cluster_size,
                    truncation=False,
                    return_token_type_ids=False,
            )
            #IGNORE WHITE SPACE IF EXISTS
            clusters['input_ids'] = [ x for x in clusters['input_ids'] if x !=3]
            cluster_ids = clusters['input_ids'][:-1]
            target_cluster_labels = ([0])*cluster_size
            for cls_ in example.cluster.split(" "):
                target_cluster_labels[cluster_dict[cls_]] = 1

        else:
            cluster_ids, target_cluster_labels, cluster_attention_mask = [],[],[]

        input_ids = inputs["input_ids"]
        input_ids = input_ids #+ [tokenizer.eos_token_id]
        input_length = len(input_ids) -1
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        #TODO check here to add clusters
        if example.text_b is not None:
            target_ids = []
            special_token = tokenizer.convert_tokens_to_ids("<p>") #space
            targets = tokenizer.encode_plus(
                    example.text_b, None,
                    add_special_tokens=True,
                    max_length=max_length_decoder-1,
                    truncation=True,
                    return_token_type_ids=False,
                )

            target_ids = targets['input_ids']

        else:
            target_ids = []

        target_input_ids = [pad_token] + target_ids
        target_output_ids = target_ids + [tokenizer.eos_token_id]
        target_attention_mask = [1 if mask_padding_with_zero else 0] * len(target_input_ids)
        ## for interpret functions --> I just deleted them since we don't use it
        #ref_ids = [pad_token] * len(input_ids[0:-1]) + [input_ids[-1]]

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        target_padding_length = max_length_decoder - len(target_input_ids)
        #TODO Currently not coding for pad_on_left option!
        if pad_on_left:

            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] \
                    * padding_length) + attention_mask

            target_input_ids = ([pad_token] * target_padding_length) \
                                + target_input_ids
            target_attention_mask = ([0 if mask_padding_with_zero else 1] \
                    * target_padding_length) + target_attention_mask

            target_output_ids = ([pad_token] * target_padding_length) + target_output_ids

        else:
            #Target looks same
            target_input_ids = target_input_ids + ([pad_token] * target_padding_length)
            target_attention_mask = target_attention_mask \
                    + ([0 if mask_padding_with_zero else 1] * target_padding_length)

            target_output_ids = target_output_ids + ([pad_token] * target_padding_length)
            bce_attention_mask = None
            if label_type == 'bcl':
                if 'train' in example.guid:
                    bce_attention_mask = [1 if mask_padding_with_zero else 0] * (len(input_ids)-1) + [0 if mask_padding_with_zero else 1] * (padding_length+ max_cluster_size+1)
                    input_ids = input_ids[:-1] + cluster_ids + [tokenizer.eos_token_id]
                    padding_length = padding_length + max_cluster_size - len(cluster_ids)
                    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) + [0 if mask_padding_with_zero else 1] * padding_length
                    input_ids = input_ids + ([pad_token] * padding_length)
                else:
                    input_ids = input_ids + ([pad_token] * (padding_length+max_cluster_size))
                    #MAKE NOT TO PUT 1 in the last part
                    bce_attention_mask = attention_mask[:-1] +[0 if mask_padding_with_zero else 1] * (padding_length+max_cluster_size+1)
                    attention_mask = attention_mask + [0 if mask_padding_with_zero else 1] * (padding_length+max_cluster_size)

            elif label_type == 'mcg':
                if style_text_a == 'MultiCluster':
                    input_ids = input_ids + ([pad_token] * (padding_length+max_cluster_size))
                    attention_mask = attention_mask + [0 if mask_padding_with_zero else 1] * (padding_length+max_cluster_size)
                elif style_text_a == 'MultiLabel':
                    input_ids = input_ids[:-1] + cluster_ids + [tokenizer.eos_token_id]
                    padding_length = padding_length + max_cluster_size - len(cluster_ids)
                    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) + [0 if mask_padding_with_zero else 1] * padding_length
                    input_ids = input_ids + ([pad_token] * padding_length)
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask \
                                + ([0 if mask_padding_with_zero else 1] * padding_length)

            #padding cluster_ids
            cluster_length = len(cluster_ids)
            cluster_ids = cluster_ids + [0] * (max_cluster_size - len(cluster_ids))

        ## Assert
        if label_type in ('bcl','mcg'):
            assert len(input_ids) == max_length+max_cluster_size, "Error with input length {} vs {}".format(len(input_ids), max_length+max_cluster_size)
            assert len(attention_mask) == max_length+max_cluster_size, "Error with input length {} vs {}".format(len(attention_mask), max_length+max_cluster_size)

        else:
            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(target_input_ids) == max_length_decoder, "Error with input length {} vs {}".format(len(target_input_ids), max_length_decoder)
        assert len(target_attention_mask) == max_length_decoder, "Error with input length {} vs {}".format(len(target_attention_mask), max_length_decoder)

        label_id = example.label
        if verbose and ex_index < 2:
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            target_tokens = tokenizer.convert_ids_to_tokens(target_input_ids)
            logger.info("*** Example ***")
            logger.info("guid: %s" % (highlight(example.guid)))
            logger.info("tokens: %s" % highlight(" ".join([str(x) for x in tokens])))
            logger.info("target tokens: %s" % highlight(" ".join([str(x) for x in target_tokens])))

            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("target input_ids: %s" % " ".join([str(x) for x in target_input_ids]))
            logger.info("target output_ids: %s" % " ".join([str(x) for x in target_output_ids]))

            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("target attention_mask: %s" % " ".join([str(x) for x in target_attention_mask]))

            # logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: {} ({})".format(str(highlight(label_id)), str(highlight(example.label))))
        features.append(
                T5ClusterInputFeatures(input_ids=input_ids,
                        attention_mask=attention_mask,
                        target_cluster_labels=target_cluster_labels,
                        target_input_ids=target_input_ids,
                        target_attention_mask=target_attention_mask,
                        target_output_ids=target_output_ids,
                        cluster_id=cluster_ids,
                        cluster_attention_mask=cluster_attention_mask,
                        label_id=label_id,
                        bce_attention_mask=bce_attention_mask,
                        input_length=[input_length,cluster_length]))
    return features

def add_loss_weight(train_dataset,epoch,out_len):
    dataset_to_list = list(zip(*train_dataset))
    #even numbers are MultiCluster
    cluster_idxs = [idx for idx in range(0,len(train_dataset),2)]
    label_idxs = [idx for idx in range(1,len(train_dataset),2)]
    #odd numbers are MultiLabel
    new_inputs = []
    #Should rearrange TensorData for additions
    for idx in range(len(dataset_to_list)):
        new_inputs.append(torch.stack(dataset_to_list[idx]))

    weight_features = [[1.0]*out_len if x in cluster_idxs else [1.0/epoch]*out_len for x in range(len(train_dataset))]
    weight_features = torch.tensor([f for f in weight_features], dtype=torch.float)
    return TensorDataset(new_inputs[0],
                    new_inputs[1],
                    new_inputs[2],
                    new_inputs[3],
                    new_inputs[4],
                    new_inputs[5],
                    new_inputs[6],
                    weight_features) # last one is loss weight

def add_cluster_noise(args, tokenizer, train_dataset):
    rand_seed = np.random.randint(1,1000)
    np.random.seed(rand_seed)
    print(rand_seed)
    mcl_token = tokenizer.encode("MultiCluster")[0]
    cluster_size = int(args.cluster_type.split("_")[-1])
    cluster_tokens = tokenizer.encode(" ".join(['<c'+str(x)+">" for x in range(cluster_size)]))[:-1]
    for ex_idx in range(len(train_dataset)):
        example = train_dataset[ex_idx]
        if mcl_token == example[0][0]:
            continue
        input_length = example[-1][0]
        cluster_length = example[-1][1].item()
        cluster_id = deepcopy(example[-2][:cluster_length])
        binom = np.random.binomial(cluster_length, args.cluster_noise_ratio,cluster_length)
        if sum(binom)>0:
            rand_cluster = np.random.choice(cluster_tokens, cluster_length)
            for idx in range(cluster_length):
                if binom[idx] == 1:
                    cluster_id[idx] = rand_cluster[idx]
        train_dataset[ex_idx][0][input_length:input_length+cluster_length] = cluster_id
    return train_dataset

# CHECK FOR bcl and mcg
def get_new_input_with_cluster(batch,cls_idxs,tokenizer,cluster_list,label_type,input_cluster_length):
    new_input_ids = []
    new_attention_masks = []
    new_labels = []
    new_clusters = []
    pad_id = tokenizer.pad_token_id
    if label_type=='mcg':
        #cls_idxs not needed, cluster_list --> cluster_id
        #cluster id shouldn't include </s> --> replace it to pad
        cluster_list[cluster_list==1] = 0
        for b_, c_ in zip(batch,cluster_list):
            if c_[0] == 0:
                cluster_id = c_[1:input_cluster_length+1]
            else:
                cluster_id = c_[:input_cluster_length]

            cluster_id = [x for x in cluster_id.tolist() if x != 3 and x !=0]
            cluster_attention_mask = [0 if x==0 else 1 for x in cluster_id]
            #TODO TASK TO MultiLabel (now it is MANUAL)
            prefix = tokenizer.encode("MultiLabel")[:1]
            padding_length = len(b_[0]) - sum(b_[1]) - len(cluster_id) + 1
            new_input_id = prefix + b_[0].tolist()[1:sum(b_[1])-1] + cluster_id[:-1] + [tokenizer.eos_token_id] + [pad_id]*padding_length
            new_input_ids.append(new_input_id)
            new_attention_mask = [1 if pad_id==0 else 0] * (len(new_input_id) - padding_length) + [0 if pad_id==0 else 1] * padding_length
            new_attention_masks.append(new_attention_mask)
            new_labels.append(b_[3])

            cluster_padding_length = input_cluster_length - len(cluster_id)
            cluster_id = cluster_id + ([tokenizer.pad_token_id]*cluster_padding_length)
            new_clusters.append(cluster_id)


    elif label_type=='bcl':
        for c_, b_ in zip(cls_idxs, batch):
            cluster_id = tokenizer.encode(' '.join(cluster_list[c_]))[:-1]
            #delete space
            cluster_id = [x for x in cluster_id if x != 3]
            if len(cluster_id) > input_cluster_length:
                cluster_id=cluster_id[:input_cluster_length]
            new_input_id = b_[0].tolist()[:sum(b_[1])-1] + cluster_id + [tokenizer.eos_token_id]
            padding_length = len(b_[0]) - len(new_input_id)
            new_input_id = new_input_id + [pad_id] * padding_length
            new_input_ids.append(new_input_id)

            cluster_padding_length = input_cluster_length - len(cluster_id)
            cluster_id = cluster_id + ([tokenizer.pad_token_id]*cluster_padding_length)
            new_clusters.append(cluster_id)

            new_attention_mask = [1 if pad_id==0 else 0] * (len(new_input_id) - padding_length) + [0 if pad_id==0 else 1] * padding_length
            new_attention_masks.append(new_attention_mask)
            new_labels.append(b_[3])

    new_input_ids = torch.tensor([f for f in new_input_ids], dtype=torch.long)
    new_attention_masks = torch.tensor([f for f in new_attention_masks], dtype=torch.long)
    new_labels = torch.stack(new_labels)
    new_clusters = torch.tensor([f for f in new_clusters], dtype=torch.long)
    return TensorDataset(new_input_ids,
                        new_attention_masks,
                        new_labels,
                        new_clusters)


def shuffle_target_ids(args, mode, tokenizer, loaded_dataset):
    task = args.task_name
    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()

    targets = processor.get_targets(args.data_dir,args.decode_order,args.label_type,
                                        mode, args.cluster_type)
    targets = targets
    features_targets = convert_targets_to_features(
            targets,
            tokenizer,
            pad_on_left=False, #bool(args.model_type in ['xlnet']),
            pad_token = tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id)

    tgt_in_ids = torch.tensor([f.target_input_ids for f in features_targets], dtype=torch.long)
    tgt_out_ids = torch.tensor([f.target_output_ids for f in features_targets], dtype=torch.long)
    num_of_ids = len(loaded_dataset[0])
    new_inputs = []
    dataset_to_list = list(zip(*loaded_dataset))
    #THIS IS MANUALLY DECIDED FROM THE ORDER IN MAIN FUNCTION
    for idx in range(0,num_of_ids):
        if idx == 2:
            new_inputs.append(tgt_in_ids)
        elif idx == 3:
            new_inputs.append(tgt_out_ids)
        else:
            new_inputs.append(torch.stack(dataset_to_list[idx]))

    #THIS IS WAY TOO MUCH UGLY SINCE I JUST SELF COUNTED POSSIBLE NUMBER OF INPUTS...
    #print((args.decode_order,args.label_type))
    #for i_ in range(0,10):
    #    print(tokenizer.decode(loaded_dataset[i_][2]))
    #    print(targets[i_])
    if num_of_ids == 5:
        return TensorDataset(new_inputs[0],
                        new_inputs[1],
                        new_inputs[2],
                        new_inputs[3],
                        new_inputs[4])
    elif num_of_ids == 6:
        return TensorDataset(new_inputs[0],
                        new_inputs[1],
                        new_inputs[2],
                        new_inputs[3],
                        new_inputs[4],
                        new_inputs[5])
    elif num_of_ids == 7:
        return TensorDataset(new_inputs[0],
                        new_inputs[1],
                        new_inputs[2],
                        new_inputs[3],
                        new_inputs[4],
                        new_inputs[5],
                        new_inputs[6])

    elif num_of_ids == 8:
        return TensorDataset(new_inputs[0],
                        new_inputs[1],
                        new_inputs[2],
                        new_inputs[3],
                        new_inputs[4],
                        new_inputs[5],
                        new_inputs[6],
                        new_inputs[7])

    elif num_of_ids == 9:
        return TensorDataset(new_inputs[0],
                        new_iputs[1],
                        new_inputs[2],
                        new_inputs[3],
                        new_inputs[4],
                        new_inputs[5],
                        new_inputs[6],
                        new_inputs[7],
                        new_inputs[8])

    else:
        print("SOMETHING WEIRED HAPPENDED TO YOUR NUM OF INPUTS IN DATASET")
    return loaded_dataset

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the l sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break

        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)

    label_set = list(set(labels))
    new_labels, new_preds =  [], []
    unmatched_label_prediction_cnt = 0
    for l,p in zip(labels,preds):
        if p not in label_set:
            unmatched_label_prediction_cnt += 1
        else:
            new_preds.append(p)
            new_labels.append(l)
    if unmatched_label_prediction_cnt > 0:
        # from pdb import set_trace; set_trace()
        f1 = f1_score(y_true=new_labels, y_pred=new_preds, average='macro')
    else:
        f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "unmatched_label_prediction_cnt": unmatched_label_prediction_cnt,
        "cnt": len(preds)
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }




class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()


    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_text(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            lines = []
            for line in f:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line.strip())
            return lines

class EURProcessor(DataProcessor):
    def load_file_name(self, mode,file_type):
        if file_type=='texts':
            return '%s_raw_%s.txt' %(mode,file_type)
        else:
            return '%s_%s.txt' %(mode,file_type)

    def get_targets(self, data_dir, decode_order, label_type=None, mode='train', cluster_type='kmeans'):
        label_name = self.load_file_name(mode,'labels')
        labels = self._read_text(os.path.join(data_dir,label_name))
        c_list, clusters = [], []

        #if label_type != 'base':
        if 'base' not in label_type:
            c_list = self._read_text(os.path.join(data_dir,'cluster','cluster_%s_%s'\
                                                    %(mode,cluster_type)))

            # CLUSTER INFO (distinct only)
            clusters=[]
            for cls_ in c_list:
                clusters.append(' '.join(sorted(list(set(['<'+x+'>' for x in cls_.split(" ")])))))
        #DO SHUFFLE IN HERE!
        if decode_order=='shuffle':
            import random
            shuffle_seed = random.choices(range(0,len(labels)),k=len(labels))
            labels_, c_list_, clusters_ = [], [], []
            for idx, seed in enumerate(shuffle_seed):
                lab_ = labels[idx].split(" ")
                #Shuffling
                random.Random(seed).shuffle(lab_)
                #Appending
                labels_.append(' '.join(lab_))

                if len(c_list) > 0:
                    cls_ = c_list[idx].split(" ")
                    cls2_ = clusters[idx].split(" ")
                    random.Random(seed).shuffle(cls_)
                    random.Random(seed).shuffle(cls2_)
                    c_list_.append(' '.join(cls_))
                    clusters_.append(' '.join(cls2_))

            labels = labels_
            c_list = c_list_
            clusters = clusters_
        # INPUT LABEL+CLUSTER FORMAT
        labels_new = []
        if label_type=='mcg':
            labels_new = []
            for lbl, cls in zip(labels, clusters):
                #Manually doubled line for 2 tasks
                labels_new.append(cls)
                labels_new.append(lbl)

        else: #Just Label info (None)
            labels_new = labels

        return labels_new

    #TODO: option decode_order
    def get_train_examples(self, data_dir, decode_order,mode, label_type=None, cluster_type='kmeans', train_data_dir=None):
        label_name = self.load_file_name(mode,'labels')
        text_name = self.load_file_name(mode,'texts')
        if train_data_dir is not None:
            texts = self._read_text(os.path.join(train_data_dir, text_name))
            labels = self._read_text(os.path.join(train_data_dir,label_name))
        else:
            texts = self._read_text(os.path.join(data_dir, text_name))
            labels = self._read_text(os.path.join(data_dir,label_name))
        if 'freq' in decode_order:
            lab_ = [item for sublist in [x.split(" ") for x in labels] \
                                        for item in sublist]
            from collections import Counter
            srt = [x for x,y in Counter(lab_).most_common(len(set(lab_)))]#[::-1]
            if 'backward' in decode_order:
                srt = srt[::-1]
            srt = {b: i for i, b in enumerate(srt)}
            labels = [' '.join(sorted(x.split(" "), key=lambda i: srt[i])) for x in labels]
        elif decode_order=='shuffle':
            #random order
            import random
            labels = [' '.join(random.sample(x.split(" "),len(x.split(" "))))
                    for x in labels]
        elif decode_order=='alphabet':
            #alphabetical order
            labels = [' '.join(sorted(x.split(" "))) for x in labels]

        #Load label-cluster map and make clusters above based on labels
        label_map = self._read_text(os.path.join(data_dir,"label_map.txt"))
        #if label_type !='base':
        if 'base' not in label_type:
            if train_data_dir is not None:
                label_cls_map = self._read_text(os.path.join(train_data_dir,'cluster',"label_map_cluster_%s.txt")%(cluster_type))
            else:
                label_cls_map = self._read_text(os.path.join(data_dir,'cluster',"label_map_cluster_%s.txt")%(cluster_type))

            label_to_cls_dict = dict()
            for x,y in zip(label_map, label_cls_map):
                label_to_cls_dict[x] = y
            c_list = [' '.join([label_to_cls_dict[x] for x in y.split(" ")]) for y in labels]
            # CLUSTER INFO (distinct only)
            clusters=[]
            c_list_natsorted = [' '.join(sorted(x.split(), key = x.split().count,reverse = True)) for x in c_list]
            # distinct clusters are sorted by freq in that example
            for cls_ in c_list_natsorted:
                clusters.append(' '.join(list(dict.fromkeys(['<'+x+'>' for x in cls_.split(" ")]))))
        else:
            clusters = []

        # INPUT LABEL+CLUSTER FORMAT
        labels_new = []
        if label_type=='mcg':
            texts_new, labels_new, clusters_new = [],[],[]
            for txt, lbl, cls in zip(texts, labels, clusters):
                #Manually doubled line for 2 tasks
                texts_new.append("<MultiCluster> " + txt)
                labels_new.append(cls)
                clusters_new.append(cls)
                texts_new.append("<MultiLabel> " + txt)
                labels_new.append(lbl)
                clusters_new.append(cls)
            texts = texts_new
            clusters = clusters_new

        else: #Just Label info (None)
            labels_new = labels

        return self._create_parallel_examples(
                texts,
                labels_new,
                clusters, #distinct clusters per each example
                "train")

    def get_dev_examples(self, data_dir, decode_order,mode, label_type=None, cluster_type='kmeans',train_data_dir=None):
        label_name = self.load_file_name(mode,'labels')
        text_name = self.load_file_name(mode,'texts')
        texts = self._read_text(os.path.join(data_dir, text_name))
        labels = self._read_text(os.path.join(data_dir,label_name))
        if 'freq' in decode_order:
            lab_ = [item for sublist in [x.split(" ") for x in labels] \
                                        for item in sublist]
            from collections import Counter
            srt = [x for x,y in Counter(lab_).most_common(len(set(lab_)))]#[::-1]
            if 'backward' in decode_order:
                srt = srt[::-1]
            srt = {b: i for i, b in enumerate(srt)}
            labels = [' '.join(sorted(x.split(" "), key=lambda i: srt[i])) for x in labels]
        elif decode_order=='shuffle':
            # random order
            import random
            labels = [' '.join(random.sample(x.split(" "),len(x.split(" "))))
                    for x in labels]
        elif decode_order=='alphabet':
            #alphabetical order
            labels = [' '.join(sorted(x.split(" "))) for x in labels]

        #Load label-cluster map and make clusters above based on labels
        label_map = self._read_text(os.path.join(data_dir,"label_map.txt"))
        #if label_type !='base':
        if 'base' not in label_type:
            if train_data_dir is not None:
                label_cls_map = self._read_text(os.path.join(train_data_dir,'cluster',"label_map_cluster_%s.txt")%(cluster_type))
            else:
                label_cls_map = self._read_text(os.path.join(data_dir,'cluster',"label_map_cluster_%s.txt")%(cluster_type))

            label_to_cls_dict = dict()
            for x,y in zip(label_map, label_cls_map):
                label_to_cls_dict[x] = y
            c_list = [' '.join([label_to_cls_dict[x] for x in y.split(" ")]) for y in labels]
            # CLUSTER INFO (distinct only)
            clusters=[]
            for cls_ in c_list:
                clusters.append(' '.join(sorted(list(set(['<'+x+'>' for x in cls_.split(" ")])))))
        else:
            clusters = []


        # INPUT LABEL+CLUSTER FORMAT
        labels_new = []
        if label_type=='mcg':
            texts_new, labels_new, clusters_new = [],[],[]
            for txt, lbl, cls in zip(texts, labels, clusters):
                #For evaluation, final label should be label not cluster even its task is multicluster
                texts_new.append("<MultiCluster> " + txt)
                labels_new.append(lbl) ## THIS IS lbl not cls
                clusters_new.append(cls)
                # for test set we are not going to make MultiLabel example since it will be generated AFTER predicting cluster
            texts = texts_new
            clusters = clusters_new

        else: #Just Label info (None)
            labels_new = labels

        return self._create_parallel_examples(
                texts,
                labels_new,
                clusters,
            "dev")

    def get_test_examples(self, data_dir, decode_order,mode, label_type=None, cluster_type='kmeans',train_data_dir=None):
        label_name = self.load_file_name(mode,'labels')
        text_name = self.load_file_name(mode,'texts')
        texts = self._read_text(os.path.join(data_dir, text_name))
        labels = self._read_text(os.path.join(data_dir,label_name))
        if 'freq' in decode_order:
            lab_ = [item for sublist in [x.split(" ") for x in labels] \
                                        for item in sublist]
            from collections import Counter
            srt = [x for x,y in Counter(lab_).most_common(len(set(lab_)))]#[::-1]
            if 'backward' in decode_order:
                srt = srt[::-1]
            srt = {b: i for i, b in enumerate(srt)}
            labels = [' '.join(sorted(x.split(" "), key=lambda i: srt[i])) for x in labels]
        elif decode_order=='shuffle':
            # random order
            import random
            labels = [' '.join(random.sample(x.split(" "),len(x.split(" "))))
                    for x in labels]
        elif decode_order=='alphabet':
            #alphabetical order
            labels = [' '.join(sorted(x.split(" "))) for x in labels]

        #Load label-cluster map and make clusters above based on labele
        label_map = self._read_text(os.path.join(data_dir,"label_map.txt"))
        #if label_type !='base':
        if 'base' not in label_type:
            if train_data_dir is not None:
                label_cls_map = self._read_text(os.path.join(train_data_dir,'cluster',"label_map_cluster_%s.txt")%(cluster_type))
            else:
                label_cls_map = self._read_text(os.path.join(data_dir,'cluster',"label_map_cluster_%s.txt")%(cluster_type))

            label_to_cls_dict = dict()
            for x,y in zip(label_map, label_cls_map):
                label_to_cls_dict[x] = y
            c_list = [' '.join([label_to_cls_dict[x] for x in y.split(" ")]) for y in labels]
            # CLUSTER INFO (distinct only)
            clusters=[]
            for cls_ in c_list:
                clusters.append(' '.join(sorted(list(set(['<'+x+'>' for x in cls_.split(" ")])))))
        else:
            clusters = []


        # INPUT LABEL+CLUSTER FORMAT
        labels_new = []
        #mcg
        if label_type=='mcg':
            texts_new, labels_new, clusters_new = [],[],[]
            for txt, lbl, cls in zip(texts, labels, clusters):
                #Manually doubled line for 2 tasks
                #For evaluation, final label should be label not cluster even its task is multicluster
                texts_new.append("<MultiCluster> " + txt)
                labels_new.append(lbl) ## This is lbl not cls
                clusters_new.append(cls)
            texts = texts_new
            clusters = clusters_new

        else: #Just Label info (None)
            labels_new = labels
        return self._create_parallel_examples(
                texts,
                labels_new,
                clusters,
            "test")

#    def get_labels(self):
#        """See base class."""
#        return ["None"]

    def get_clusters(self,data_dir,cluster_type='kmeans'):
        """Get the cluster lists"""
        clusters = []
        with open(os.path.join(data_dir, 'cluster','label_map_cluster_%s.txt'%(cluster_type)),'r') as f:
            for line in f:
                clusters.append(' '.join(['<'+x+'>' for x in line.replace("\n","").split(" ")]))

        clusters = sorted(clusters, key = clusters.count, reverse = True)
        return list(dict.fromkeys(clusters))

    def get_labels(self,data_dir):
        """See base class."""
        # In order to add labels to the vocabulary
        labels = []
        with open(os.path.join(data_dir, "label_map.txt"),'r') as f:
            for line in f:
                labels.append(line.replace("\n",""))

        labels =sorted(list(set(labels)))
        return labels

    def _create_parallel_examples(self, src_lines, tgt_lines,cluster_lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        assert len(src_lines) == len(tgt_lines)

        if len(cluster_lines) >0:
            for (i, (src,tgt,cst)) in enumerate(zip(src_lines, tgt_lines,cluster_lines)):
                guid = "%s-%s" % (set_type, i)
                try:
                    text_a = src
                    text_b = tgt
                    label = None
                    cluster= cst
                except Exception as e:
                    print(i,src,tgt,cst)
                    from pdb import set_trace; set_trace()

                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,cluster=cluster))
        else:
            for (i, (src,tgt)) in enumerate(zip(src_lines, tgt_lines)):
                guid = "%s-%s" % (set_type, i)
                try:
                    text_a = src
                    text_b = tgt
                    label = None
                    cluster= None
                except Exception as e:
                    print(i,src,tgt)
                    from pdb import set_trace; set_trace()

                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,cluster=cluster))
        return examples

class AAPDProcessor(EURProcessor):
    def __init__(self):
        super().__init__()

#Wiki is just a child class of EURProcessor
class WikiProcessor(EURProcessor):
    def __init__(self):
        super().__init__()


processors = {
    "wiki10-31k": WikiProcessor,
    "eur-lex": EURProcessor,
    "aapd": AAPDProcessor,
    "amazoncat-13k": WikiProcessor,
    "wiki-500k": WikiProcessor,
    "amazon-670k": WikiProcessor,
    "amazon-3m": WikiProcessor,

}

output_modes = {
    "wiki10-31k": "transfer",
    "eur-lex": "transfer",
    "aapd": "transfer",
    "amazoncat-13k": "transfer",
    "wiki-500k": "transfer",
    "amazon-670k": "transfer",
    "amazon-3m": "transfer",
}

