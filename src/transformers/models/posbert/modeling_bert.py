# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_bert import PosBertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google-bert/bert-base-uncased"
_CONFIG_FOR_DOC = "PosBertConfig"

# TokenClassification docstring
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)
_TOKEN_CLASS_EXPECTED_LOSS = 0.01

# QuestionAnswering docstring
_CHECKPOINT_FOR_QA = "deepset/bert-base-cased-squad2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 7.41
_QA_TARGET_START_INDEX = 14
_QA_TARGET_END_INDEX = 15

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "textattack/bert-base-uncased-yelp-polarity"
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
_SEQ_CLASS_EXPECTED_LOSS = 0.01


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model

class SemBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        print("initializing SemBertEmbeddings")
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        print("forward SemBertEmbeddings")
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class PosBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        print("initializing PosBertEmbeddings")
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.pos_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.pos_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.pos_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        print("forward PosBertEmbeddings")
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PosBertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, layer_idx=None):
        super().__init__()
        print("initializing PosBertSelfAttention")
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        if config.pos_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The pos size ({config.pos_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int((config.hidden_size + config.pos_size) / config.num_attention_heads)
        self.sem_attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.pos_attention_head_size = int(config.pos_size / config.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sem_all_head_size = self.num_attention_heads * self.sem_attention_head_size
        self.pos_all_head_size = self.num_attention_heads * self.pos_attention_head_size
       
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if config.fully_indep_sem_pos : 
            # Queries and keys separated
            self.sem_query = nn.Linear(config.hidden_size, self.sem_all_head_size)
            self.pos_query = nn.Linear(config.pos_size, self.pos_all_head_size)

            self.sem_key = nn.Linear(config.hidden_size, self.sem_all_head_size)
            self.pos_key = nn.Linear(config.pos_size, self.pos_all_head_size)
        
        else :
            self.query = nn.Linear(config.pos_size + config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.pos_size + config.hidden_size, self.all_head_size)

        # Value is separated (position / semantic)
        self.sem_value = nn.Linear(config.hidden_size, self.sem_all_head_size)
        self.pos_value = nn.Linear(config.pos_size, self.pos_all_head_size)

        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    
    def transpose_for_scores(self, x: torch.Tensor, type_emb: Optional[str] = None) -> torch.Tensor:
        """
        Args: 
            x: tenseur de taille [batch_size, seq_len, hidden_size]
        Returns: 
            x: tenseur de taille [batch_size, num_heads, seq_len, head_size]
        """
        if type_emb == "pos" :
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.pos_attention_head_size)
        elif type_emb == "sem" :
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.sem_attention_head_size)
        elif type_emb == "all":
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        else :
            raise ValueError (f"{type_emb} is not a valid type of embedding")
        
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        pos_hidden_states: torch.Tensor,
        sem_hidden_states: torch.Tensor,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        batch_size, seq_length, _ = pos_hidden_states.size()

        print("hidden_size", self.config.hidden_size)
        print("positional head dim", pos_hidden_states.shape)
        print("semantic head dim", sem_hidden_states.shape)
        # Séparation des parties positionnelle et sémantique
        # p = hidden_states[:, :, :self.positional_dim]  # [batch, seq_len, positional_dim]
        # s = hidden_states[:, :, self.positional_dim:]  # [batch, seq_len, semantic_dim]


        if self.config.fully_indep_sem_pos : 
            print("compute Q")
            query_pos_layer = self.pos_query(pos_hidden_states)
            query_sem_layer = self.sem_query(sem_hidden_states)

            print("compute K")
            key_pos_layer = self.pos_key(pos_hidden_states)
            key_sem_layer = self.sem_key(sem_hidden_states)

        else :
            hidden_states = torch.concat([pos_hidden_states, sem_hidden_states], dim=-1)
            print("compute full")
            query_layer = self.query(hidden_states)
            key_layer = self.query(hidden_states)
            print(query_layer.shape, key_layer.shape)
            
        print("computy V")
        value_pos_layer = self.pos_value(pos_hidden_states)
        value_sem_layer = self.sem_value(sem_hidden_states)


        # mixed_query_layer = self.query(hidden_states)
        # mixed_key_layer = self.key(hidden_states)
        


        def compute_attention_probs_and_context(query, key, value, attention_mask, qk_type_emb = "pos", v_type_emb = None) :
            
            if v_type_emb is None :
                v_type_emb=qk_type_emb
            
            v_head_size = self.pos_attention_head_size if v_type_emb=="pos" else self.sem_attention_head_size 

            query_Tr = self.transpose_for_scores(query, qk_type_emb) # [batchsize, num_head, seq_len, head_dim]
            key_Tr = self.transpose_for_scores(key, qk_type_emb)
            value_Tr = value.view(batch_size, seq_length, self.num_attention_heads, v_head_size)
            value_Tr = value_Tr.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]

            attention_scores = torch.matmul(query_Tr, key_Tr.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(v_head_size)

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)

            context = torch.matmul(
                attention_probs, # [batch, num_attention_heads, seq_len, seq_len]
                value_Tr # [batch, num_attention_heads, seq_len, head_dim]
            )

            context = context.permute(0, 2, 1, 3).contiguous() # [batch, seqlength, num_head, head_dim]
            new_context_shape = context.size()[:-2] + (v_head_size * self.num_attention_heads,)
            context = context.view(new_context_shape)

            return attention_probs, context

        if self.config.fully_indep_sem_pos : 
            # POSITIONAL ATTENTION
            print("computing pos context")
            pos_attention_probs, pos_context = compute_attention_probs_and_context(query_pos_layer, key_pos_layer, value_pos_layer, attention_mask, "pos")


            # SEMANTIC ATTENTION
            print("computing sem context")
            sem_attention_probs, sem_context = compute_attention_probs_and_context(query_sem_layer, key_sem_layer, value_sem_layer, attention_mask, "sem")
        
        else :
            # POSITIONAL ATTENTION
            print("computing mixed pos context")
            pos_attention_probs, pos_context = compute_attention_probs_and_context(query_layer, key_layer, value_pos_layer, attention_mask, "all", "pos")

            # SEMANTIC ATTENTION
            print("computing mixed sem context")
            sem_attention_probs, sem_context = compute_attention_probs_and_context(query_layer, key_layer, value_sem_layer, attention_mask, "all", "sem")
       
        # Concaténation des résultats
        # context_layer = torch.cat([pos_context, sem_context], dim=-1) # [batch, num_heads, seq_len, position_head_dim + semantic_head_dim = attention_head_size]

        # outputs = (context_layer, pos_attention_probs, pos_attention_scores) if output_attentions else (context_layer,)
        return (pos_context, sem_context, pos_attention_probs, sem_attention_probs)



class PosBertAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        print("initializing PosBertAttention")
        self.self = PosBertSelfAttention(config, layer_idx)
        self.pos_output = PosBertSelfOutput(config)
        self.sem_output = SemBertSelfOutput(config)
        self.num_attention_heads = config.num_attention_heads
        self.pruned_heads = set()
        self.layer_idx = layer_idx

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self, 
        pos_hidden_states,
        sem_hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None, 
        output_attentions=False
    ): 
        pos_context, sem_context, pos_attention_probs, sem_attention_probs = self.self(
            pos_hidden_states,
            sem_hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions=True,  # Forcé à True pour avoir les scores bruts
        )

        # # Vérification de la sortie de self-attention
        # assert self_outputs[0].shape == hidden_states.shape, (
        #     f"Mismatch in shapes: self_outputs[0] shape {self_outputs[0].shape} "
        #     f"does not match hidden_states shape {hidden_states.shape}"
        # )

        print("out of self attention")
        print("pos context shape", pos_context.shape)
        print("sem context shape", sem_context.shape)
        pos_attention_output = self.pos_output(pos_context, pos_hidden_states)
        sem_attention_output = self.sem_output(sem_context, sem_hidden_states)
        

        # Vérification de la sortie finale
        assert pos_attention_output.shape == pos_hidden_states.shape, (
            f"Mismatch in shapes: attention_output shape {pos_attention_output.shape} "
            f"does not match hidden_states shape {pos_hidden_states.shape}"
        )

        assert sem_attention_output.shape == sem_hidden_states.shape, (
            f"Mismatch in shapes: attention_output shape {sem_attention_output.shape} "
            f"does not match hidden_states shape {sem_hidden_states.shape}"
        )

        outputs = (pos_attention_output, sem_attention_output, pos_attention_probs, sem_attention_probs)  # add attentions if we output them (# Les attentions includent maintenant les scores bruts)
        return outputs

class PosBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("initializing PosBertSOutput")
        # Dimensions de l'espace d'origine
        # Dimension positionnelle = d_head
        self.pos_dim = config.pos_size
        
        # L'espace intermédiaire est supposé être de dimension 4 * hidden_size.
        # On le divise en deux branches :
        #   branche positionnelle de dimension 4 * pos_dim
        #   branche sémantique de dimension 4 * sem_dim
        self.intermediate_pos_dim = 4 * self.pos_dim
        # Vérification implicite : self.intermediate_pos_dim + self.intermediate_sem_dim == config.intermediate_size
        
        # Deux projections linéaires distinctes sur chaque branche
        self.dense = nn.Linear(self.intermediate_pos_dim, self.pos_dim)
        
        self.LayerNorm = nn.LayerNorm(self.pos_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # hidden_states est de dimension [batch, seq_len, intermediate_size] avec
        # intermediate_size = 4 * hidden_size = 4 * pos_dim + 4 * sem_dim
        # Séparation en deux branches le long de la dernière dimension :

        # Application des projections linéaires distinctes
        pos_out = self.dense(hidden_states)  # [batch, seq_len, pos_dim]
        
        # Concaténation des deux branches pour obtenir un vecteur de dimension hidden_size
        pos_out = self.dropout(pos_out)
        output = self.LayerNorm(pos_out + input_tensor)
        return output
    
class SemBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("initializing SemBertSOutput")
        # Dimensions de l'espace d'origine
        # Dimension positionnelle = d_head
        self.sem_dim = config.hidden_size
        
        # L'espace intermédiaire est supposé être de dimension 4 * hidden_size.
        # On le divise en deux branches :
        #   branche positionnelle de dimension 4 * sem_dim
        #   branche sémantique de dimension 4 * sem_dim
        self.intermediate_sem_dim = 4 * self.sem_dim
        # Vérification implicite : self.intermediate_sem_dim + self.intermediate_sem_dim == config.intermediate_size
        
        # Deux projections linéaires distinctes sur chaque branche
        self.dense = nn.Linear(self.intermediate_sem_dim, self.sem_dim)
        
        self.LayerNorm = nn.LayerNorm(self.sem_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # hidden_states est de dimension [batch, seq_len, intermediate_size] avec
        # intermediate_size = 4 * hidden_size = 4 * pos_dim + 4 * sem_dim
        # Séparation en deux branches le long de la dernière dimension :

        # Application des projections linéaires distinctes
        sem_out = self.dense(hidden_states)  # [batch, seq_len, sem_dim]
        
        # Concaténation des deux branches pour obtenir un vecteur de dimension hidden_size
        sem_out = self.dropout(sem_out)
        output = self.LayerNorm(sem_out + input_tensor)
        return output


class SemBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("initializing SemBertSelfOutput")
        # Détermination des dimensions pour la branche positionnelle et sémantique
        self.sem_dim = config.hidden_size

        # Projections linéaires distinctes pour chaque branche :
        # On projette de sem_dim à sem_dim et de sem_dim à sem_dim
        self.dense = nn.Linear(self.sem_dim, self.sem_dim)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, sem_hidden_states: torch.Tensor, sem_input_tensor: torch.Tensor) -> torch.Tensor:
        print("in sembertselfoutput forward")
        # Application des projections linéaires sur chacune des branches
        sem_output = self.dense(sem_hidden_states)            # [batch, seq_len, pos_dim]

        # Reconstruction de la sortie en concaténant les deux branches
        sem_output = self.dropout(sem_output)
        sem_output = self.LayerNorm(sem_output + sem_input_tensor)
        return sem_output
    
class PosBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Détermination des dimensions pour la branche positionnelle et sémantique
        print("initializing PosBertSelfOutput")
        self.pos_dim = config.pos_size

        # Projections linéaires distinctes pour chaque branche :
        # On projette de pos_dim à pos_dim et de sem_dim à sem_dim
        self.dense = nn.Linear(self.pos_dim, self.pos_dim)

        self.LayerNorm = nn.LayerNorm(config.pos_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pos_hidden_states: torch.Tensor, pos_input_tensor: torch.Tensor) -> torch.Tensor:
        print("in posbertselfoutput forward")
        # Application des projections linéaires sur chacune des branches
        pos_output = self.dense(pos_hidden_states)            # [batch, seq_len, pos_dim]
        print("pos_output", pos_output.shape)
        print("pos_input_tensor", pos_input_tensor.shape)
        # Reconstruction de la sortie en concaténant les deux branches
        pos_output = self.dropout(pos_output)
        pos_output = self.LayerNorm(pos_output + pos_input_tensor)
        return pos_output
    


class PosBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # On décompose l'espace d'entrée (hidden_size) en deux parties :
        # - d_pos est défini comme d_head = hidden_size // num_attention_heads
        # - d_sem est le complément : hidden_size - d_pos
        print("initializing PosBertIntermediate")
        self.pos_dim = config.pos_size

        # On veut projeter dans un espace intermédiaire de dimension 4 * hidden_size.
        # On répartit cet espace en deux branches :
        # - branche positionnelle de dimension 4 * pos_dim
        # - branche sémantique de dimension 4 * sem_dim
        self.pos_intermediate_dim = 4 * self.pos_dim

        # Les couches de projection sont appliquées sur les parties correspondantes de l'entrée.
        self.dense = nn.Linear(self.pos_dim, self.pos_intermediate_dim)
        print("IIIIIIIIIIIIIIIIIII", self.dense, self.pos_dim, self.pos_intermediate_dim)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        # Application des projections linéaires sur chaque branche
        print("IIIIIIIIIIIIIIIIIII", self.dense, self.pos_dim, self.pos_intermediate_dim, hidden_states.shape)
        pos_out = self.dense(hidden_states)   # [batch, seq_len, 4 * pos_dim]

        # Application de l'activation sur chaque branche
        pos_out = self.intermediate_act_fn(pos_out)

        # Regroupement par concaténation le long de la dernière dimension :
        # La sortie finale aura une dimension de 4 * (pos_dim + sem_dim) = 4 * hidden_size
        return pos_out
        
class SemBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # On décompose l'espace d'entrée (hidden_size) en deux parties :
        # - d_pos est défini comme d_head = hidden_size // num_attention_heads
        # - d_sem est le complément : hidden_size - d_pos
        print("initializing PosBertIntermediate")
        self.sem_dim = config.hidden_size

        # On veut projeter dans un espace intermédiaire de dimension 4 * hidden_size.
        # On répartit cet espace en deux branches :
        # - branche positionnelle de dimension 4 * pos_sem
        # - branche sémantique de dimension 4 * sem_dim
        self.sem_intermediate_dim = 4 * self.sem_dim

        # Les couches de projection sont appliquées sur les parties correspondantes de l'entrée.
        self.dense = nn.Linear(self.sem_dim, self.sem_intermediate_dim)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        try:
            # Application des projections linéaires sur chaque branche
            sem_out = self.dense(hidden_states)   # [batch, seq_len, 4 * sem_dim]

            # Application de l'activation sur chaque branche
            sem_out = self.intermediate_act_fn(sem_out)

            # Regroupement par concaténation le long de la dernière dimension :
            # La sortie finale aura une dimension de 4 * (pos_dim + sem_dim) = 4 * hidden_size
            return sem_out
        
        except Exception as e:
            print(f"Error in PosBertIntermediate.forward: {e}")
            print(f"hidden_states type: {type(hidden_states)}, shape: {hidden_states.shape if hasattr(hidden_states, 'shape') else 'N/A'}")
            print(f"self.dense_pos.weight type: {type(self.dense_pos.weight)}, shape: {self.dense_pos.weight.shape if hasattr(self.dense_pos.weight, 'shape') else 'N/A'}")
            print(f"self.dense_sem.weight type: {type(self.dense_sem.weight)}, shape: {self.dense_sem.weight.shape if hasattr(self.dense_sem.weight, 'shape') else 'N/A'}")
            raise



class PosBertLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        print("initializing PosBertLayer")
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PosBertAttention(config, layer_idx)
        self.pos_intermediate = PosBertIntermediate(config)
        self.sem_intermediate = SemBertIntermediate(config)
        self.pos_output = PosBertOutput(config)
        self.sem_output = SemBertOutput(config)

    def forward(
        self,
        pos_hidden_states,
        sem_hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        
        try:
            pos_attention_output, sem_attention_output, pos_attention_probs, sem_attention_probs = self.attention(
                pos_hidden_states,
                sem_hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
            )
            # attention_output = self_attention_outputs[0]
            outputs = (pos_attention_probs, sem_attention_probs) # add self attentions if we output attention weights

            # Version avec débogage sans chunking pour isoler le problème

            pos_intermediate_output = self.pos_intermediate(pos_attention_output)
            sem_intermediate_output = self.sem_intermediate(sem_attention_output)
            print("after intermediate pos", pos_intermediate_output.shape)
            print("after intermediate sem", sem_intermediate_output.shape)
            pos_layer_output = self.pos_output(pos_intermediate_output, pos_attention_output)
            sem_layer_output = self.sem_output(sem_intermediate_output, sem_attention_output)

            # Commentez la version avec chunking pour tester sans
            # layer_output = apply_chunking_to_forward(
            #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            # )
            
            outputs = (pos_layer_output,sem_layer_output) + outputs
            return outputs
            
        except Exception as e:
            print(f"Error in PosBertLayer.forward: {e}")
            import traceback
            traceback.print_exc()
            raise

    def feed_forward_chunk(self, attention_output):
        try:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output
        except Exception as e:
            print(f"Error in feed_forward_chunk: {e}")
            import traceback
            traceback.print_exc()
            raise


class PosBertEncoder(nn.Module):
    def __init__(self, config):
        print("initializing PosBertEncoder")
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([
            PosBertLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
        self,
        pos_hidden_states: torch.Tensor,
        sem_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                hidden_states = torch.cat([pos_hidden_states, sem_hidden_states], dim=-1)
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                layer_outputs = self._gradient_checkpointing_func(
                    create_custom_forward(layer_module),
                    pos_hidden_states,
                    sem_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    pos_hidden_states,
                    sem_hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                pos_hidden_states, sem_hidden_states = layer_outputs[0], layer_outputs[1]

    
            if output_attentions:
                all_self_attentions = all_self_attentions + layer_outputs[2:]

        hidden_states = torch.cat([pos_hidden_states, sem_hidden_states], dim=-1)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        print("ending the encoder!")
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
    
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class PosBertPooler(nn.Module):
    def __init__(self, config):
        print("initializing PosBertPooler")
        super().__init__()
        # Ici, nous divisons hidden_size en deux parties.
        # Vous pouvez ajuster ce partage. Par exemple, ici on prend la moitié pour le positionnel
        # et l'autre moitié pour le sémantique.
        self.pos_dim = config.position_dimensions
        self.sem_dim = config.hidden_size - self.pos_dim
        
        # Deux transformations linéaires distinctes
        self.dense_pos = nn.Linear(self.pos_dim, self.pos_dim)
        self.dense_sem = nn.Linear(self.sem_dim, self.sem_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # On pool en prenant le token [CLS] (premier token)
        first_token_tensor = hidden_states[:, 0]  # [batch, hidden_size]
        # On sépare le vecteur en deux parties : positionnel et sémantique
        pos_part = first_token_tensor[:, :self.pos_dim]  # [batch, pos_dim]
        sem_part = first_token_tensor[:, self.pos_dim:]  # [batch, sem_dim]
        
        # Transformation linéaire distincte pour chaque partie
        pos_transformed = self.dense_pos(pos_part)  # [batch, pos_dim]
        sem_transformed = self.dense_sem(sem_part)  # [batch, sem_dim]
        
        # Concaténation des deux branches
        pooled_output = torch.cat([pos_transformed, sem_transformed], dim=-1)  # [batch, hidden_size]
        pooled_output = self.activation(pooled_output)
        return pooled_output



class PosBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pos_size + config.hidden_size, config.pos_size + config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.pos_size + config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        print("3", hidden_states.shape)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        print("4", hidden_states.shape)
        return hidden_states


class PosBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = PosBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.pos_size + config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        print("2", hidden_states.shape)
        hidden_states = self.transform(hidden_states)
        print("5", hidden_states.shape)
        print(self.decoder.weight.shape)
        hidden_states = self.decoder(hidden_states)
        print("6", hidden_states.shape)
        return hidden_states


class PosBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = PosBertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        print("1", sequence_output.shape)
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class PosBertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class PosBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = PosBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PosBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PosBertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
class PosBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`PosBertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


BERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PosBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`or `(batch_size, sequence_length, target_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class PosBertModel(PosBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    """

    _no_split_modules = ["PosBertEmbeddings", "PosBertLayer"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        print("initializing PosBertModel")
        print(config)
        self.sem_embeddings = SemBertEmbeddings(config)
        self.pos_embeddings = PosBertEmbeddings(config)
        print("set up sem and pos embed", self.sem_embeddings, self.pos_embeddings)
        self.encoder = PosBertEncoder(config)

        self.pooler = PosBertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        print("got input embeddings", self.pos_embeddings.position_embeddings)
        return self.pos_embeddings.position_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, target_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        
        """
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)


        print("input ids shape", input_ids.shape)
        pos_embedding_output = self.pos_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        sem_embedding_output = self.sem_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        print("got pos embeddings", pos_embedding_output.shape)
        print("got sem embeddings", sem_embedding_output.shape)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and head_mask is None
            and not output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # raise NotImplementedError
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]

            extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                attention_mask, pos_embedding_output.dtype, tgt_len=seq_length
            )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            pos_embedding_output,
            sem_embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
)
class PosBertForPreTraining(PosBertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = PosBertModel(config)
        self.cls = PosBertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=PosBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], PosBertForPreTrainingOutput]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, *optional*, defaults to `{}`):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, PosBertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> model = PosBertForPreTraining.from_pretrained("google-bert/bert-base-uncased")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return PosBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a `language modeling` head on top for CLM fine-tuning.""", BERT_START_DOCSTRING
)
class PosBertLMHeadModel(PosBertPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = PosBertModel(config, add_pooling_layer=False)
        self.cls = PosBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **loss_kwargs,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            lm_loss = self.loss_function(prediction_scores, labels, self.config.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class PosBertForMaskedLM(PosBertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        print("starting posbert for masked LM")
        self.bert = PosBertModel(config, add_pooling_layer=False)
        self.cls = PosBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        print("CALLED GET OUTPUT EMBEDDINGS")
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        print("CALLED SET OUTPUT EMBEDDINGS")
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.88,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        print("sequence output shape", sequence_output.shape)
        print("uuuuuuh", self.cls.predictions.decoder.weight.shape)
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top.""",
    BERT_START_DOCSTRING,
)
class PosBertForNextSentencePrediction(PosBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = PosBertModel(config)
        self.cls = PosBertOnlyNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, PosBertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> model = PosBertForNextSentencePrediction.from_pretrained("google-bert/bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class PosBertForSequenceClassification(PosBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = PosBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BERT_START_DOCSTRING,
)
class PosBertForMultipleChoice(PosBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = PosBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class PosBertForTokenClassification(PosBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = PosBertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
class PosBertForQuestionAnswering(PosBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = PosBertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_QA,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
        expected_output=_QA_EXPECTED_OUTPUT,
        expected_loss=_QA_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "PosBertForMaskedLM",
    "PosBertForMultipleChoice",
    "PosBertForNextSentencePrediction",
    "PosBertForPreTraining",
    "PosBertForQuestionAnswering",
    "PosBertForSequenceClassification",
    "PosBertForTokenClassification",
    "PosBertLayer",
    "PosBertLMHeadModel",
    "PosBertModel",
    "PosBertPreTrainedModel",
    "load_tf_weights_in_bert",
]