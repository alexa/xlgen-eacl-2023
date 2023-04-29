import os
import shutil
import coloredlogs, logging
from colorama import Fore,Style
from tqdm import tqdm, trange
from collections import Counter
from typing import Any, List, Dict, Tuple, Optional
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from scipy.sparse import load_npz, save_npz, csr_matrix
from utils import highlight #compute_metrics,
from utils import shuffle_target_ids, get_new_input_with_cluster, add_cluster_noise, add_loss_weight
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(
        self,
        args: List[Any],
        model: Any,
        tokenizer,
        train_dataset: Optional[TensorDataset] = None,
        dev_dataset: Optional[TensorDataset] = None,
        test_dataset: Optional[TensorDataset] = None,
    ) -> None:

        self.args, self.model_args, self.data_args = args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = self.args.ignore_index  # 0 #self.tokenizer.pad_token_id #

        self.model = model
        # GPU or CPU
        self.device = (
            "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        )
        self.model.to(self.device)
        self.model.parallelize()

    def get_representation(self):
        #Load train dataset
        #SHOULD BE SEQUENTIALLY ORDERED
        train_sampler = SequentialSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        train_representations = []
        self.model.eval()
        iterator = tqdm(train_dataloader, desc="Generating Train", position=0, leave=True)
        for step, batch in enumerate(iterator):
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                with torch.no_grad():
                    inputs = {'input_ids':      batch[0],
                                'attention_mask': batch[1]}
                    hidden_states=self.model(**inputs,output_encoder_embed_only=True)
                    hidden_states = hidden_states.detach().cpu()
                    attention_mask = batch[1].detach().cpu()
                    hidden_states = hidden_states.mul(attention_mask.unsqueeze(-1))
                    pooled_hidden_states = hidden_states.sum(dim=1)
                    pooled_hidden_states = torch.div(pooled_hidden_states,torch.sum(attention_mask,dim=1).unsqueeze(-1))
                    train_representations.extend(pooled_hidden_states.tolist())
        #save representations as np
        logger.info("Save train t5 representations")
        np.save(os.path.join(self.data_args.data_dir,"X.trn.%s.npy"%(self.model_args.model_name)),train_representations)

        #DOIT FOR THE TEST AND DEV
        dev_sampler = SequentialSampler(self.dev_dataset)
        dev_dataloader = DataLoader(self.dev_dataset, sampler=dev_sampler, batch_size=self.args.train_batch_size)
        dev_representations = []
        self.model.eval()
        iterator = tqdm(dev_dataloader, desc="Generating Dev", position=0, leave=True)
        for step, batch in enumerate(iterator):
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                with torch.no_grad():
                    inputs = {'input_ids':      batch[0],
                                'attention_mask': batch[1]}
                    hidden_states=self.model(**inputs,output_encoder_embed_only=True)
                    hidden_states = hidden_states.detach().cpu()
                    attention_mask = batch[1].detach().cpu()
                    hidden_states = hidden_states.mul(attention_mask.unsqueeze(-1))
                    pooled_hidden_states = hidden_states.sum(dim=1)
                    pooled_hidden_states = torch.div(pooled_hidden_states,torch.sum(attention_mask,dim=1).unsqueeze(-1))

                    pooled_hidden_states = torch.mean(hidden_states,dim=1)
                    dev_representations.extend(pooled_hidden_states.detach().cpu().tolist())
        #save representations as np
        logger.info("Save dev t5 representations")
        np.save(os.path.join(self.data_args.data_dir,"X.dev.%s.npy"%(self.model_args.model_name)),dev_representations)

        #DOIT FOR THE TEST AND DEV
        test_sampler = SequentialSampler(self.test_dataset)
        test_dataloader = DataLoader(self.test_dataset, sampler=test_sampler, batch_size=self.args.train_batch_size)
        test_representations = []
        self.model.eval()
        iterator = tqdm(test_dataloader, desc="Generating Test", position=0, leave=True)
        for step, batch in enumerate(iterator):
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                with torch.no_grad():
                    inputs = {'input_ids':      batch[0],
                            'attention_mask': batch[1]}
                    hidden_states=self.model(**inputs,output_encoder_embed_only=True)
                    hidden_states = hidden_states.detach().cpu()
                    attention_mask = batch[1].detach().cpu()
                    hidden_states = hidden_states.mul(attention_mask.unsqueeze(-1))
                    pooled_hidden_states = hidden_states.sum(dim=1)
                    pooled_hidden_states = torch.div(pooled_hidden_states,torch.sum(attention_mask,dim=1).unsqueeze(-1))

                    pooled_hidden_states = torch.mean(hidden_states,dim=1)
                    test_representations.extend(pooled_hidden_states.detach().cpu().tolist())
        #save representations as np
        logger.info("Save test t5 representations")
        np.save(os.path.join(self.data_args.data_dir,"X.tst.%s.npy"%(self.model_args.model_name)),test_representations)


    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = {}".format( highlight(len(self.train_dataset))))
        logger.info("  Num Epochs = {}".format( highlight(self.args.num_train_epochs)))
        logger.info("  Total train batch size = {}".format( highlight(self.args.train_batch_size)))
        logger.info("  Gradient Accumulation steps = {}".format( highlight(self.args.gradient_accumulation_steps)))
        logger.info("  Total optimization steps = {}".format( highlight(t_total)))
        logger.info("  Logging steps = {}".format( highlight(self.args.logging_steps)))
        logger.info("  Save steps = {}".format( highlight(self.args.save_steps)))

        global_step = 0
        tr_loss = 0.0
        dev_score_history, dev_step_history = [], []
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        #keep epoch num for saving
        epoch=0
        for _ in train_iterator:
            epoch+=1
            if self.data_args.decode_order=='shuffle':
                self.train_dataset = shuffle_target_ids(self.data_args,'train',self.tokenizer,self.train_dataset)
                train_sampler = RandomSampler(self.train_dataset)
                train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
            #TODO TEST IT IF ITS TRAINING INSTACES ARE RANDOMLY ORDERED PER EVERY EPOCH
            if self.data_args.label_type !='base' \
                and self.data_args.cluster_noise_ratio>0:
                    self.train_dataset = add_cluster_noise(self.data_args, self.tokenizer, self.train_dataset)
                    train_sampler = RandomSampler(self.train_dataset)
                    train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

            #train size for MultiCluster will be 1/epoch --> decreasing
            if self.data_args.label_type == 'decoder-sep' and self.args.annealing:
                #add loss weight in here and update it every epoch
                self.train_dataset = add_loss_weight(self.train_dataset,epoch,self.args.max_out_seq_len)
                train_sampler = RandomSampler(self.train_dataset)
                train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

            #update logging_steps for evaluation
            #self.args.logging_steps = int(len(train_dataloader)/5)
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
            self.model.decoder_sep_annealing=False #Just in case
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                if self.data_args.label_type=='bcl' :
                    lambda_denom = 1
                    if self.args.annealing:
                        lambda_denom = epoch
                    inputs = {'input_ids':      batch[0],
                            'attention_mask': batch[1],
                            'decoder_input_ids': batch[2],
                            'decoder_attention_mask': batch[4],
                            'labels':         batch[3],
                            'bce_attention_mask': batch[5],
                            'cluster_labels': batch[6],
                            'lambda_denom': lambda_denom, } #bce loss --> 1/lambda_denom

                elif self.data_args.use_cluster:
                    lambda_denom = 1
                    if self.args.annealing:
                        lambda_denom = epoch
                    #MAKE SURE WE HAVE DIFFERENT DATA ORDER FROM BCE_SEP
                    inputs = {'input_ids':      batch[0],
                            'attention_mask': batch[1],
                            'decoder_input_ids': batch[2],
                            'decoder_attention_mask': batch[4],
                            'labels':         batch[3],
                            'cluster_labels': batch[5],
                            'lambda_denom': lambda_denom, } #bce loss --> 1/lambda_denom

                elif self.data_args.label_type=='decoder-sep' and self.args.annealing:
                    #TODO input with weight
                    inputs= {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'decoder_input_ids': batch[2],
                          'decoder_attention_mask': batch[4],
                          'labels':         batch[3],
                          'weight_loss': batch[7]}


                else:
                    inputs= {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'decoder_input_ids': batch[2],
                          'decoder_attention_mask': batch[4],
                          'labels':         batch[3],}

                outputs = self.model(**inputs)
                loss = outputs[0]
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                epoch_iterator.set_description("loss={:.2f} lr={:.1e}".format(
                    tr_loss / (global_step+1), scheduler.get_last_lr()[0]
                ))

                if (step + 1) % (self.args.gradient_accumulation_steps) == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                # dev out for 5 times in every epoch
                #if step==0:
                #    results = self.evaluate("dev", save_output=False)

                #    result_to_save = {'model':self.model_args.model_name_or_path,
                #                   'global_step':global_step }

                #    dev_score = 0.0

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

        #always save last epoch
        self.save_model()
        #Just save model BEFORE end --> now we get the wrong(?) output when we reload model
        return global_step, tr_loss / global_step


    def evaluate(self, mode, verbose=False, save_result=False, save_output=False):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = {}".format( highlight(len(dataset))))
        logger.info("  Batch size = {}".format( highlight(self.args.eval_batch_size)))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds_ids = None
        out_label_ids = None
        input_ids = None
        cluster_list = None
        self.model.eval()
        iterator = tqdm(eval_dataloader, desc="Evaluating")
        for batch in iterator:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                #original input
                inputs = {'input_ids':      batch[0], \
                            'attention_mask': batch[1], \
                            'decoder_start_token_id': 0, \
                                #'pad_token_id':0, \
                                'bos_token_id':0, \
                                #'eos_token_id':1, \
                                'max_length':self.args.max_out_seq_len, \
                                'min_length':1, \
                                'do_sample':self.args.do_sample, \
                                'early_stopping':self.args.early_stopping, \
                                'num_beams':self.args.num_beams, \
                                'temperature':self.args.temperature, \
                                'top_k':self.args.top_k, \
                                'top_p':self.args.top_p, \
                                'repetition_penalty':self.args.repetition_penalty, \
                                'pad_token_id':self.tokenizer.pad_token_id, \
                                'eos_token_id':self.tokenizer.eos_token_id, \
                                'length_penalty':self.args.length_penalty, \
                                'no_repeat_ngram_size':self.args.no_repeat_ngram_size, \
                                'num_return_sequences':self.args.num_return_sequences, \
                                   }
                labels = batch[3]

                if self.data_args.use_cluster:
                    if self.data_args.label_type=='bcl':
                        inputs_cl = {'input_ids':      batch[0], \
                                   'attention_mask': batch[1], \
                                   'decoder_input_ids': batch[2], \
                                   'bce_attention_mask': batch[5]
                                }
                    else:
                        inputs_cl = {'input_ids':      batch[0], \
                                   'attention_mask': batch[1], \
                                   'decoder_input_ids': batch[2], \
                                }

                    prob_cl = self.model(**inputs_cl,output_cluster_predict_only=True)
                    prob_cl_sigmoid = torch.sigmoid(prob_cl)

                    if cluster_list is None:
                        from utils import processors
                        processor = processors[self.data_args.task_name.lower()]()
                        #FOR PU Setup new data path for cluster should be loaded.
                        if self.data_args.train_data_dir is not None:
                            cluster_dir = self.data_args.train_data_dir
                        else:
                            cluster_dir = self.data_args.data_dir

                        cluster_list = processor.get_clusters(cluster_dir,\
                                                    self.data_args.cluster_type)
                        cluster_list = np.array(cluster_list)

                if self.data_args.label_type == 'decoder-sep':
                    #First, get predicted cluster_id for generate input
                    inputs['top_k'] = 1
                    inputs['no_repeat_ngram_size'] = 1
                    inputs['num_beams'] = 1
                    inputs['num_return_sequences'] = 1
                    cl_outputs = self.model.generate(**inputs)
                    cl_outputs = cl_outputs.to("cpu")
                    new_batch = tuple(t.to('cpu') for t in batch)
                    new_batch = list(zip(*new_batch))
                    tmp_dataset = get_new_input_with_cluster(new_batch,
                                            [],
                                            self.tokenizer,
                                            cl_outputs,
                                            self.data_args.label_type,
                                            self.args.input_cluster_length)
                    #load sampler and dataloader
                    tmp_sampler = SequentialSampler(tmp_dataset)
                    tmp_dataloader = DataLoader(tmp_dataset,
                                            sampler=tmp_sampler,
                                            batch_size = self.args.eval_batch_size)
                    tmp_stack = None
                    outputs = []
                    for tmp_batch in tmp_dataloader:
                        tmp_batch = tuple(t.to(self.device) for t in tmp_batch)
                        tmp_inputs = {'input_ids':      tmp_batch[0], \
                                    'attention_mask': tmp_batch[1], \
                                    'decoder_start_token_id': 0, \
                                    #'pad_token_id':0, \
                                    'bos_token_id':0, \
                                    #'eos_token_id':1, \
                                    'max_length':self.args.max_out_seq_len, \
                                    'min_length':1, \
                                    'do_sample':self.args.do_sample, \
                                    'early_stopping':self.args.early_stopping, \
                                    'num_beams':self.args.num_beams, \
                                    'temperature':self.args.temperature, \
                                    'top_k':self.args.top_k, \
                                    'top_p':self.args.top_p, \
                                    'repetition_penalty':self.args.repetition_penalty, \
                                    'pad_token_id':self.tokenizer.pad_token_id, \
                                    'eos_token_id':self.tokenizer.eos_token_id, \
                                    'length_penalty':self.args.length_penalty, \
                                    'no_repeat_ngram_size':self.args.no_repeat_ngram_size, \
                                    'num_return_sequences':self.args.num_return_sequences, \
                                       }

                        tmp_labels = tmp_batch[2]
                        tmp_outputs = self.model.generate(**tmp_inputs)
                        if self.args.num_return_sequences > 1:
                            redim = (int(tmp_outputs.shape[0]/self.args.num_return_sequences),int(tmp_outputs.shape[1]*self.args.num_return_sequences))
                            tmp_outputs = tmp_outputs.reshape(redim)

                        #incorporate output and cluster for this example
                        #append to out
                        outputs = torch.cat((tmp_batch[3],tmp_outputs),1)
                elif self.data_args.label_type=='bcl':
                    num_cls_selected= (prob_cl_sigmoid>self.data_args.bce_thold).sum(axis=1)
                    #set min num
                    num_cls_selected = torch.where(num_cls_selected < 1, 1, num_cls_selected).tolist()
                    # Loop for single example
                    #bring batch to cpu before modify them
                    new_batch = tuple(t.to('cpu') for t in batch)
                    new_batch = list(zip(*new_batch))
                    outputs = []
                    cls_idxs = [prob_cl_sigmoid[idx].topk(x)[1].detach().cpu().tolist() for idx, x in enumerate(num_cls_selected)]
                    tmp_dataset = get_new_input_with_cluster(new_batch,
                                                    cls_idxs,
                                                    self.tokenizer,
                                                    cluster_list,
                                                    self.data_args.label_type,
                                                    self.args.input_cluster_length)

                    #load sampler and dataloader
                    tmp_sampler = SequentialSampler(tmp_dataset)
                    tmp_dataloader = DataLoader(tmp_dataset, sampler=tmp_sampler, batch_size = self.args.eval_batch_size)
                    tmp_stack = None
                    for tmp_batch in tmp_dataloader:
                        tmp_batch = tuple(t.to(self.device) for t in tmp_batch)
                        tmp_inputs = {'input_ids':      tmp_batch[0], \
                                    'attention_mask': tmp_batch[1], \
                                    'decoder_start_token_id': 0, \
                                    #'pad_token_id':0, \
                                    'bos_token_id':0, \
                                    #'eos_token_id':1, \
                                    'max_length':self.args.max_out_seq_len, \
                                    'min_length':1, \
                                    'do_sample':self.args.do_sample, \
                                    'early_stopping':self.args.early_stopping, \
                                    'num_beams':self.args.num_beams, \
                                    'temperature':self.args.temperature, \
                                    'top_k':self.args.top_k, \
                                    'top_p':self.args.top_p, \
                                    'repetition_penalty':self.args.repetition_penalty, \
                                    'pad_token_id':self.tokenizer.pad_token_id, \
                                    'eos_token_id':self.tokenizer.eos_token_id, \
                                    'length_penalty':self.args.length_penalty, \
                                    'no_repeat_ngram_size':self.args.no_repeat_ngram_size, \
                                    'num_return_sequences':self.args.num_return_sequences, \
                                       }

                        tmp_labels = tmp_batch[2]
                        tmp_outputs = self.model.generate(**tmp_inputs)
                        #incorporate output and cluster for this example
                        #append to out
                        if self.args.num_return_sequences > 1:
                            redim = (int(tmp_outputs.shape[0]/self.args.num_return_sequences),int(tmp_outputs.shape[1]*self.args.num_return_sequences))
                            tmp_outputs = tmp_outputs.reshape(redim)
                        outputs = torch.cat((tmp_batch[3],tmp_outputs),1)

                else:
                    outputs = self.model.generate(**inputs)

                preds_batch, labels_batch, input_batch, cl_batch = [], [], [],[]
                if 't5' in self.model_args.model_name_or_path:
                    if self.data_args.use_cluster:
                        prob_cl_sigmoid = prob_cl_sigmoid.cpu().detach().tolist()
                        for idx, cl in enumerate(prob_cl_sigmoid):
                            cl_batch.append(cl)
                    #always keep pred_batch no matter which label_types we use
                    for idx, (in_ids, pred, label) in enumerate(zip(inputs["input_ids"], outputs, labels)):
                        input_tokens = self.tokenizer.convert_ids_to_tokens(in_ids.tolist())
                        input_str = self.tokenizer.convert_tokens_to_string(input_tokens)
                        input_batch.append(input_str)

                        pred_tokens = self.tokenizer.convert_ids_to_tokens(pred.tolist())
                        pred_str = self.tokenizer.convert_tokens_to_string(pred_tokens)
                        preds_batch.append(pred_str)

                        label_tokens = self.tokenizer.convert_ids_to_tokens(label.tolist())
                        label_str = self.tokenizer.convert_tokens_to_string(label_tokens)
                        labels_batch.append(label_str)

                elif self.model_args.model_name_or_path.startswith("facebook/bart"):
                    for idx, (in_ids, pred, label) in enumerate(zip(inputs["input_ids"], outputs, labels)):
                        input_str = self.tokenizer.decode(in_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        input_batch.append(input_str)

                        pred_str = self.tokenizer.decode(pred.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        preds_batch.append(pred_str)

                        label_str = self.tokenizer.decode(label.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        labels_batch.append(label_str)

                        cl_batch.append(cl)
                else:
                    print("Wrong model name at inference",self.model_args.model_name_or_path)
                    import sys
                    sys.exit(1)


                if preds_ids is None:
                    input_ids = input_batch
                    preds_ids = preds_batch
                    out_label_ids = labels_batch
                    if self.data_args.use_cluster:
                        cl_ids = cl_batch
                else:
                    input_ids = np.append(input_ids, input_batch)
                    preds_ids = np.append(preds_ids, preds_batch)
                    out_label_ids = np.append(out_label_ids, labels_batch)
                    if self.data_args.use_cluster:
                        cl_ids = np.concatenate((cl_ids, cl_batch))

            # preds_ids = preds
            # eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        if out_label_ids is not None:
            for idx, (i, l, p) in enumerate(zip(input_ids, preds_ids, out_label_ids)):
                if idx>=3:
                    break
                print('----------{}------------'.format(idx))
                print(highlight(i))
                print(highlight(l))
                print(highlight(p))
                print('----------------------')

            # if save_result:
                # result = compute_metrics(eval_task, preds_ids, out_label_ids)
            #     results.update(result)

            # save results to file
            if save_output:
                if not os.path.exists(self.data_args.eval_output_dir):
                    os.makedirs(self.data_args.eval_output_dir)

                if self.data_args.eval_dataset is not None:
                    basename_eval_dataset = os.path.basename(self.data_args.eval_dataset).replace('.txt','')
                    output_pred_file = os.path.join(self.data_args.eval_output_dir, "eval_preds_{}.txt".format(basename_eval_dataset))
                else:
                    output_pred_file = os.path.join(self.data_args.eval_output_dir, "eval_preds_test.txt")

                with open(output_pred_file, "w") as writer:
                    logger.info("***** Save prediction outputs: {} ******".format(highlight(output_pred_file)))
                    for idx, (i, l, p) in enumerate(zip(input_ids, preds_ids, out_label_ids)):
                        writer.write("{}\n".format(l.replace("<pad>","").replace("<"," <")))

                # save cluster prob as well
                if self.data_args.use_cluster:
                    save_npz(os.path.join(self.data_args.eval_output_dir,"CL_P.npz"),csr_matrix(cl_ids))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        output_dir = os.path.join(self.args.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", highlight(output_dir))

        #Save Tokenizer as well
        self.tokenizer.save_pretrained(output_dir)

    def copy_best_model(self, best_dir_name='checkpoint_best'):
        output_dir = self.args.output_dir
        best_dir = os.path.join(self.args.output_dir, best_dir_name)
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        os.makedirs(best_dir)

        files = (file for file in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, file)))
        for file in files:
            shutil.copy(os.path.join(output_dir,file), best_dir)
