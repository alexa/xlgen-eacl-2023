import argparse
import glob
import pprint
import numpy as np
import os
from collections import defaultdict
import random
import pdb
import math
import coloredlogs, logging
import torch

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    TrainingArguments
  )




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Original model name for cached data. Default is None"
        },
    )



    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    #additional model name for correct cached example output
    model_name: Optional[str] = field(
            default=None, metadata={"help": "Original model name before finetuening"}
    )



@dataclass
class TrainingArguments(TrainingArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    ignore_index: Optional[int] = field(
        default=0,
        metadata={
            "help": "Specifies a target value that is ignored and does not contribute to the input gradient"},
    )
    slot_loss_coef: Optional[float] = field(
        default=1.0,
        metadata={"help": "Coeffcient for the slot loss"},
    )
    use_crf: bool = field(
        default=False, metadata={"help": "Wehther to use CRF"}
    )
    slot_pad_label: Optional[str] = field(
        default="PAD", metadata={"help": "Pad token for slot label pad (to be ignore when calculate loss)"}
    )

    dropout_rate: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Dropout for fully-connected layers"},
    )

    use_pos: bool = field(
        default=False, metadata={"help": "Wehther to use POS embedding or not"})
    use_np: bool = field(
        default=False, metadata={"help": "Wehther to use NP embedding or not"})
    use_vp: bool = field(
        default=False, metadata={"help": "Wehther to use VP embedding or not"})
    use_entity: bool = field(
        default=False, metadata={"help": "Wehther to use Entity embedding or not"})
    use_acronym: bool = field(
        default=False, metadata={"help": "Wehther to use Acronym embedding or not"})

    use_heuristic: bool = field(
        default=False, metadata={"help": "Wehther to use heuristic filters or not"})


    do_ensemble: bool = field(
        default=False, metadata={"help": "Wehther to use model ensemble or not"})


    # # Beam Search
    max_out_seq_len: Optional[int] = field(
      default=200,
      metadata={"help": "tbw"},
    )
    do_sample: bool = field(
        default=False, metadata={"help": ""})
    early_stopping: bool = field(
        default=True, metadata={"help": ""})
    num_beams: Optional[int] = field(
      default=1,metadata={"help": "TBW"})
    temperature: Optional[float] = field(
      default=1.0,metadata={"help": "TBW"})
    top_k: Optional[int] = field(
      default=50,metadata={"help": "TBW"})
    top_p: Optional[float] = field(
      default=1.0,metadata={"help": "TBW"})
    repetition_penalty: Optional[float] = field(
      default=1.0,metadata={"help": "TBW"})
    length_penalty: Optional[float] = field(
      default=1.0,metadata={"help": "TBW"})
    no_repeat_ngram_size: Optional[int] = field(
      default=0,metadata={"help": "TBW"})
    num_return_sequences: Optional[int] = field(
      default=1,metadata={"help": "TBW"})

    ## number of clusters in input ids if bce training
    input_cluster_length:Optional[int] = field(
      default=0,
      metadata={"help": "tbw"},
    )

    get_rep: bool = field(
     default=False, metadata={"help": "get t5 model encoder representation for the future"}
   )

    ## update weights of cluster training
    annealing: bool = field(
        default=False, metadata={"help": "reduce weights from cluster training as 1/epoch"}
    )


    ## save model for every epoch; default is saving every 5 epoch
    save_all_epoch: bool = field(
        default=False, metadata={"help": "reduce weights from cluster training as 1/epoch"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
      default=None,
      metadata={"help": "The name of the task to train selected in the list: "},
    )
    # eval_data_file: Optional[str] = field(
      # default=None,
      # metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    # )
    overwrite_cache: bool = field(
      default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    task: Optional[str] = field(
      default=None,
      metadata={"help": "The name of the task to train"},
    )
    kfold: Optional[int] = field(
      default=-1,
      metadata={"help": "TBW"},
    )
    data_dir: Optional[str] = field(
      default='./data',
      metadata={"help": "The input data dir"},
    )
    intent_label_file: Optional[str] = field(
      default='intent_label.txt',
      metadata={"help": "Intent label file"},
    )
    slot_label_file: Optional[str] = field(
      default='slot_label.txt',
      metadata={"help": "Slot label file"},
    )
    pos_label_file: Optional[str] = field(
      default='pos_label.txt',
      metadata={"help": "POS label file"},
    )

    max_seq_len: Optional[int] = field(
      default=500,
      metadata={"help": "TBW"},
    )

    do_lower_case: bool = field(
        default=False, metadata={"help": "Set this flag if you are using an uncased model."}
    )

    eval_dataset: Optional[str] = field(
      default=None,
      metadata={"help": "Additional input dataset"},
    )

    dev_limit: Optional[int] = field(
      default=1,
      metadata={"help": "TBW"},
    )

    eval_output_dir: Optional[str] = field(
      default="./",
      metadata={"help": ""},
    )

    #For multi label cluster info
    use_cluster: bool = field(
      default=False, metadata={"help": "Using cluster information to the training for multi label classification"}
    )

    label_type: Optional[str] = field(
            default=None,
            metadata={"help": "How to incorporate and label and cluster info in train input"}
    )

    cluster_type: Optional[str] = field(
            default='kmeans',
            metadata={"help": "Type of clustering model, default hierarchical cluster"}
    )

    #For multiple types of input datasets per each training
    cache_data_dir: Optional[str] = field(
            default='./data',
            metadata={"help": "Where the preprocessed dataset should be saved"}
    )

    ##shuffle labels while training
    decode_order: Optional[str] = field(
            default='freq_forward', metadata={"help": "Shuffling decoding labels for each epoch"})

    ## add noise injection to clusters
    cluster_noise_ratio: Optional[float] = field(
            default=0, metadata={"help": "add noise clusters in training set"})

    ## For PU case
    train_data_dir: Optional[str] = field(
      default=None,
      metadata={"help": "The input data dir"},
    )

    ## For bce thold
    bce_thold: Optional[float] = field(
            default=0.5, metadata={"help": "bce thold"})

