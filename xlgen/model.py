## Custom Model for T5 Clustering XML
## Modified code from transformers version 4.2.

###############
# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model. """


import copy
import math
import os
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from typing import Optional,Tuple

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config

from transformers.models.t5.modeling_t5 import (
        T5PreTrainedModel,
        T5Stack,
        T5_START_DOCSTRING,
        T5_INPUTS_DOCSTRING,
        PARALLELIZE_DOCSTRING,
        DEPARALLELIZE_DOCSTRING,
)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]

@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForClusterXMLGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        #additional linear layer for cluster
        #for BCE training
        self.use_cluster = config.use_cluster
        self.train_decoder_cluster = config.train_decoder_cluster
        if self.train_decoder_cluster:
            self.lm_head_dec = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.input_cluster_length= config.input_cluster_length
        if self.use_cluster:
            self.lm_head_cl = nn.Linear(config.d_model,config.cluster_size, bias=False)
            self.cluster_size = config.cluster_size

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.decoder_sep_annealing = False #Default

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True
        if self.use_cluster:
        #    self.decoder_cl = self.decoder_cl.to(self.device_map)
            self.lm_head_cl = self.lm_head_cl.to(self.decoder.first_device)
        if self.train_decoder_cluster:
            self.lm_head_dec = self.lm_head_dec.to(self.decoder.first_device)


    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

        if self.use_cluster:
        #    self.decoder_cl = self.decoder_cl.to("cpu")
            self.lm_head_cl = self.lm_head_cl.to("cpu")

        if self.train_decoder_cluster:
            self.lm_head_dec = self.lm_head_dec.to("cpu")


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        if self.train_decoder_cluster:
            self.lm_head_dec = new_embeddings

    def get_output_embeddings(self):
        if self.train_decoder_cluster:
            return self.lm_head_dec
        else:
            return self.lm_head
        #TODO maybe add self.lm_head_cl

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        cluster_output_ids=None, #for decoder-sim
        cluster_attention_mask=None, #for decoder-sim
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        cluster_labels=None, #for decoder-sim and bce-sep (have different structures)
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output_cluster_predict_only=False,
        output_encoder_embed_only=False,
        input_cluster_length = 0,
        bce_attention_mask=None,
        lambda_denom=1,
        weight_loss=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states = encoder_outputs[0]
        #For T5-representation
        if output_encoder_embed_only:
            return hidden_states

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        lm_logits_cl = None

        #FOR BCE_SEP
        if self.use_cluster and input_ids is not None:
            if bce_attention_mask is None:
                encoder_outputs_cl = self.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds,
                        head_mask=head_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,)

            else:
                encoder_outputs_cl = self.encoder(
                        input_ids=input_ids*bce_attention_mask,
                        attention_mask=bce_attention_mask,
                        inputs_embeds=inputs_embeds,
                        head_mask=head_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                )
            hidden_states_cl = encoder_outputs_cl[0]
            pooled_hidden_states = torch.mean(hidden_states_cl,dim=1)

            # update to 0 for next hidden states
            #hidden_states_cl = hidden_states_cl * bce_attention_mask.unsqueeze(-1)
            #tot_len = bce_attention_mask.sum(dim=1).unsqueeze(-1)
            #pooled_hidden_states = hidden_states_cl.sum(dim=1) / tot_len
            #Calculate logits for cluster
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head_cl = self.lm_head_cl.to(self.encoder.first_device)
                pooled_hidden_states = pooled_hidden_states.to(self.lm_head_cl.weight.device)

            lm_logits_cl = self.lm_head_cl(pooled_hidden_states)
            #As an intermediate step for evaluation, we will only output bce cluster logits
            if output_cluster_predict_only:
                return lm_logits_cl

        #labels are used like teacher-forcing for training only...
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        if past_key_values is not None:
            #TODO get attribute special token from trainer
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]
        #DO IT SEPERATELY
        if self.train_decoder_cluster:
            # Set device for model parallelism
            encoder_outputs_dec = self.encoder(
                    input_ids=input_ids*bce_attention_mask,
                    attention_mask=bce_attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
            )
            hidden_states_dec = encoder_outputs_dec[0]

            if self.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)
                hidden_states_dec = hidden_states_dec.to(self.decoder.first_device)

            decoder_outputs_dec = self.decoder(
                input_ids=cluster_labels,
                attention_mask=cluster_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states_dec,
                encoder_attention_mask=bce_attention_mask, #only use first texts
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output_dec = decoder_outputs_dec[0]
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head_dec = self.lm_head_dec.to(self.encoder.first_device)
                sequence_output_dec = sequence_output_dec.to(self.lm_head.weight.device)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)
            if self.train_decoder_cluster:
                sequence_output_dec = sequence_output_dec * (self.model_dim ** -0.5)


        lm_logits = self.lm_head(sequence_output)
        loss = None
        loss_org = None
        loss_cl = None
        loss_dec = None
        if labels is not None:
            if weight_loss is None:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss_org = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss = loss_org
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100,reduce=False)
                loss_ = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss_org = torch.mean(loss_*weight_loss.view(-1))
                loss = loss_org

            if self.use_cluster:
                loss_fct_cl = BCEWithLogitsLoss()
                loss_cl = loss_fct_cl(lm_logits_cl ,cluster_labels)
                #Control with annealing
                lambda_ = 1.0 / lambda_denom
                loss += lambda_ * loss_cl

            if self.train_decoder_cluster:
                loss_fct_dec = CrossEntropyLoss(ignore_index=-100)
                lm_logits_dec = self.lm_head_dec(sequence_output_dec)
                loss_dec = loss_fct_dec(lm_logits_dec.view(-1,lm_logits_dec.size(-1)),cluster_output_ids.view(-1))
                lambda_ = 1.0 / lambda_denom
                loss += lambda_ * loss_dec

            if self.decoder_sep_annealing:
                lambda_ = 1.0 / lambda_denom
                loss = lambda_ * loss

        if not return_dict:
            output = (lm_logits,lm_logits_cl,lm_logits_dec,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,loss_cl,loss_dec,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),)
                #reordered_layer_past_states = reordered_layer_past_states + (
                #    layer_past_state.index_select(0, beam_idx),
                #)

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


