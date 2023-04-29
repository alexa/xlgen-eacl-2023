import argparse
import glob
import os
import json
import random
from collections import defaultdict
from pprint import pprint
import numpy as np
from typing import Iterable
#from IPython.core.display import display, HTML, Image
from tqdm import tqdm, trange
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from captum.attr import LayerIntegratedGradients
from transformers import HfArgumentParser, AutoConfig, AutoTokenizer

# from utils import (compute_metrics, output_modes, processors, highlight)
# from utils import T5_convert_examples_to_features as convert_examples_to_features
# from utils import summarize_attributions, add_attributions_to_visualizer, visualize_text

import coloredlogs, logging
logger = logging.getLogger(__name__)

from utils import _set_barrier,_set_barrier_v2, _set_fp16, _set_seed, highlight, init_logger, info, processors, output_modes, convert_examples_to_features
from configuration import DataTrainingArguments, TrainingArguments,ModelArguments
from trainer import Trainer

## from models.modeling_t5 import T5ForConditionalGeneration
from model import T5ForClusterXMLGeneration

def get_max_cluster_size(args, model_name):
    task = data_args.task_name
    processor = processors[task]()
    examples_train = processor.get_train_examples(args.data_dir,args.decode_order, 'train', args.label_type,args.cluster_type,args.train_data_dir)
    examples_dev = processor.get_dev_examples(args.data_dir, args.decode_order, 'dev', args.label_type,args.cluster_type,args.train_data_dir)
    examples_test = processor.get_test_examples(args.data_dir, args.decode_order, 'test', args.label_type,args.cluster_type,args.train_data_dir)

    max_train = max( [len(list(set(x.cluster.split(" ")))) for x in examples_train])
    max_dev = max( [len(list(set(x.cluster.split(" ")))) for x in examples_dev])
    max_test = max( [len(list(set(x.cluster.split(" ")))) for x in examples_test])

    return max(max_train, max_dev, max_test)

def T5_load_and_cache_examples(args, mode, tokenizer, model_name, decode_order, evaluate=False, show_stats=True, cluster=False, max_cluster_size=0, cluster_noise_ratio=0):
    task = data_args.task_name
    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels(args.data_dir)
    if 'base' not in args.label_type:
        if args.train_data_dir is not None:
            cluster_list = processor.get_clusters(args.train_data_dir,args.cluster_type)

        else:
            cluster_list = processor.get_clusters(args.data_dir,args.cluster_type)
    else:
        cluster_list = []
    if args.eval_dataset is not None:
        basename_eval_dataset = os.path.basename(args.eval_dataset).replace('.txt','')
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(args.cache_data_dir, 'cached_{}={}_{}'.format(
            'test_input' if evaluate else 'train',
            basename_eval_dataset,
            model_name,
            str(args.max_seq_len),
            ))
    else:
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(args.cache_data_dir, 'cached_{}_{}_{}'.format(
            mode,
            model_name,
            str(args.max_seq_len),
            str(task)))

    # if there is not folder, make the directory
    if not os.path.exists(os.path.dirname(cached_features_file)):
        os.makedirs(os.path.dirname(cached_features_file))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", str(highlight(cached_features_file)))
        features = torch.load(cached_features_file)
    else:
        if args.eval_dataset is not None:
            logger.info("Creating features from dataset file at %s", str(highlight(args.eval_dataset)))
            examples = processor.get_test_input_examples(args.eval_dataset) if evaluate else processor.get_train_examples(args.data_dir,args.label_type,args.cluster_type,args.train_data_dir)
        else:
            logger.info("Creating features from dataset file at %s", str(highlight(args.data_dir)))

            if mode == "train":
                get_examples = processor.get_train_examples
            elif mode == "dev":
                get_examples = processor.get_dev_examples
            elif mode == "test":
                get_examples = processor.get_test_examples

            examples = get_examples(args.data_dir, args.decode_order, mode, args.label_type,args.cluster_type,args.train_data_dir)

        features = convert_examples_to_features(
            examples,
            tokenizer,
            task = task,
            label_list = label_list,
            cluster_list = cluster_list,
            max_length = args.max_seq_len,
            max_length_decoder=training_args.max_out_seq_len,
            output_mode = output_mode,
            pad_on_left=False, #bool(args.model_type in ['xlnet']),
            pad_token = tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            model_name=model_name,
            label_type = data_args.label_type,
            max_cluster_size = max_cluster_size)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    if evaluate and mode=="dev" and args.dev_limit > 0:
        logger.info("Limiting dev data to %d", args.dev_limit)
        features = features[:args.dev_limit]

    if show_stats and args.eval_dataset is None:
        nonzero_cnt = 0
        for feature in features:
            nonzero_cnt += (np.count_nonzero(feature.input_ids) - 2)

        # Vocab size
        vocab_dict = defaultdict(int)
        for feature in features:
            for f in feature.input_ids:
                vocab_dict[f] += 1

        stats = {
            'num_features': len(features),
            'avg_sent_len': nonzero_cnt / len(features),
            'vocab_size': len(vocab_dict),
            'input_length': args.max_seq_len + max_cluster_size,
            'target_length': training_args.max_out_seq_len, }

        for key in stats.keys():
            logger.info(" {} = {}".format(key, highlight(stats[key])))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_target_input_ids = torch.tensor([f.target_input_ids for f in features], dtype=torch.long)
    all_target_output_ids = torch.tensor([f.target_output_ids for f in features], dtype=torch.long)
    all_target_attention_mask = torch.tensor([f.target_attention_mask for f in features], dtype=torch.long)

    if args.use_cluster:
        all_target_cluster_labels = torch.tensor([f.target_cluster_labels for f in features],dtype=torch.float)

    if data_args.label_type =='bcl':
        all_bce_attention_mask = torch.tensor([f.bce_attention_mask for f in features], dtype=torch.long)
        all_input_length = torch.tensor([f.input_length for f in features])
        all_cluster_ids = torch.tensor([f.cluster_id for f in features], dtype=torch.long)

    if data_args.label_type=='mcg':
        all_cluster_ids = torch.tensor([f.cluster_id for f in features], dtype=torch.long)
        all_input_length = torch.tensor([f.input_length for f in features])


    if args.eval_dataset is None:
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        elif output_mode == "transfer":
            all_label_ids = None

    else:
        all_label_ids = None

    if all_label_ids is None:
        if args.label_type == 'bcl':
            dataset = TensorDataset(all_input_ids,
                                all_attention_mask,
                                all_target_input_ids,
                                all_target_output_ids,
                                all_target_attention_mask,
                                all_bce_attention_mask,
                                all_target_cluster_labels,
                                all_cluster_ids,
                                all_input_length)
        elif args.use_cluster:
            dataset = TensorDataset(all_input_ids,
                                all_attention_mask,
                                all_target_input_ids,
                                all_target_output_ids,
                                all_target_attention_mask,
                                all_target_cluster_labels)

        elif args.label_type in ('mcg'):
            dataset = TensorDataset(all_input_ids,
                                all_attention_mask,
                                all_target_input_ids,
                                all_target_output_ids,
                                all_target_attention_mask,
                                all_cluster_ids,
                                all_input_length)

        else:
            dataset = TensorDataset(all_input_ids,
                                all_attention_mask,
                                all_target_input_ids,
                                all_target_output_ids,
                                all_target_attention_mask)
    return dataset

def main(model_args, data_args, training_args):
    init_logger()
    # e.g., SARC_pol -> SARC in data_dir
    data_args.data_dir = data_args.data_dir.split('_')[0]
    # e.g., bert-base-uncased -> bert
    model_args.model_type = model_args.model_name_or_path.lower().split("-")[0]

    # Prepare xSLUE task
    data_args.task_name = data_args.task_name.lower()
    if data_args.task_name not in processors:
        raise ValueError("Task not found: %s" % (data_args.task_name))
    processor = processors[data_args.task_name]()
    output_mode = output_modes[data_args.task_name]
    label_list = processor.get_labels(data_args.data_dir)
    num_labels = len(label_list)
    logger.info("Processor: {}, label: ({})".format(str(highlight(processor)),str(highlight(num_labels))))

    _set_seed(training_args)
    #info(logger, training_args)

    # config, tokenizer, model
    #clusters ==> add special tokens
    if 'base' not in data_args.label_type:
        if data_args.train_data_dir is not None:
            cluster_map = processor.get_clusters(data_args.train_data_dir, \
                                                data_args.cluster_type)
        else:
            cluster_map = processor.get_clusters(data_args.data_dir,data_args.cluster_type)
        cluster_size = len(cluster_map)
    else:
        cluster_size = 0
    #add special tokens --> currently we don't do it since we are not going to generate it in the same t5 models...
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir if model_args.cache_dir else None,
    )

    config.cluster_size = cluster_size
    #Configuration setups for bce training
    max_cluster_size = 0
    if 'bcl' in data_args.label_type:
        data_args.use_cluster = True
    if data_args.label_type in ('bcl','mcg'):
        max_cluster_size =  get_max_cluster_size(data_args,
                                                 model_name=model_args.model_name)

        training_args.input_cluster_length = max_cluster_size #
        import gc
        gc.collect()

    config.train_decoder_cluster = False
    config.use_cluster = data_args.use_cluster

    #TODO we do not need input_cluster_length anymore
    config.input_cluster_length = training_args.input_cluster_length
    if os.path.exists(training_args.output_dir) and not training_args.overwrite_output_dir:
        model_name_or_path = training_args.output_dir
        model = T5ForClusterXMLGeneration.from_pretrained(model_name_or_path,config=config)
        #Just in case
        config.vocab_size = model.config.vocab_size
        logger.info(" *** Model loaded {} ***)".format(str(highlight(training_args.output_dir))))
    else:
        model_name_or_path = model_args.model_name_or_path
        model = T5ForClusterXMLGeneration.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir if model_args.cache_dir else None,
        )
        if training_args.do_train:
            logger.info(" *** Training new model from scratch ***)")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, #Directly load from the saved model
        do_lower_case=data_args.do_lower_case,
        cache_dir=model_args.cache_dir if model_args.cache_dir else None,
    )

    #Add cluster to the special tokens
    if training_args.do_train and training_args.overwrite_output_dir and 'base' not in data_args.label_type:
        #data_args.label_type !='base':
        #resize tokenizers
        #add last tokens
        cluster_map = cluster_map + [x.replace("<","</") for x in cluster_map] + ['</c>']
        #Just ADD PREFIX BY DEFAULT FOR THE FUTURE
        cluster_map = cluster_map + ['MultiCluster','MultiLabel','<p>']
        tokenizer.add_special_tokens({"additional_special_tokens":cluster_map})
        config.vocab_size = len(tokenizer)
    else:
        tokenizer.add_special_tokens({"additional_special_tokens":['<p>']})
        config.vocab_size = len(tokenizer)

    model.resize_token_embeddings(len(tokenizer))

    _set_barrier(training_args)
    _set_fp16(training_args)

    if not training_args.do_eval:
        train_dataset = T5_load_and_cache_examples(data_args, mode="train", decode_order=data_args.decode_order, model_name=model_args.model_name, tokenizer=tokenizer, evaluate=False,cluster=True,max_cluster_size=max_cluster_size,cluster_noise_ratio=data_args.cluster_noise_ratio)
    dev_dataset = T5_load_and_cache_examples(data_args, mode="dev", decode_order=data_args.decode_order, model_name=model_args.model_name,tokenizer=tokenizer, evaluate=True,cluster=False,max_cluster_size=max_cluster_size)
    test_dataset = T5_load_and_cache_examples(data_args, mode="test", decode_order=data_args.decode_order, model_name=model_args.model_name,tokenizer=tokenizer, evaluate=True,cluster=False,max_cluster_size=max_cluster_size)

    #DO NOT LOD train dataset if only eval
    if training_args.do_eval:
        # Initialize our Trainer w/t train_dataset
        trainer = Trainer(
            [training_args,model_args, data_args],
            model=model,
            tokenizer=tokenizer,
            train_dataset=None,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset
            )

        results = trainer.evaluate("test", verbose=True, save_output=True)

    else:
        # Initialize our Trainer
        trainer = Trainer(
            [training_args,model_args, data_args],
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset
            )

        if training_args.get_rep:
            trainer.get_representation()

        if training_args.do_train:
            trainer.train()





if __name__ == "__main__":
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args, data_args, training_args)

