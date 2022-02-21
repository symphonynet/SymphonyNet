from fast_transformers.builders import TransformerEncoderBuilder, RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask, LengthMask

import logging
import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)

DEFAULT_MAX_TARGET_POSITIONS = 1024

@register_model("linear_transformer_lm")
class LinearTransformerLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--embed-dim', type=int, metavar='N',
                            help='embedding dimension')
        parser.add_argument('--num-attention-heads', type=int, metavar='N',
                            help='num attention heads')
        parser.add_argument('--num-layers', type=int, metavar='N',
                            help='num layers')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability for all fully connected layers '
                                 'in the embeddings, encoder, and pooler')
        # parser.add_argument('--attention-dropout', type=float, metavar='D',
        #                     help='dropout probability for attention weights')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        return cls(LinearTransformerDecoder(args, task))


class LinearTransformerDecoder(FairseqDecoder):
    def __init__(self, args, task):

        super().__init__(task.target_dictionary)
        self.embed_dim = args.embed_dim
        self.max_seq_len = args.max_seq_len
        self.wte = nn.Embedding(len(task.target_dictionary), args.embed_dim)
        self.wpe = nn.Embedding(args.max_seq_len+1, args.embed_dim)
        self.drop = nn.Dropout(args.dropout)
        self.ln_f = nn.LayerNorm(args.embed_dim, eps=1e-6)
        
        #self.embed_tokens = Embedding(len(task.target_dictionary), args.embed_dim, self.pad_idx)
        #self.wpe = MyLearnedPositionalEmbedding(args.max_seq_len, args.embed_dim, self.pad_idx)
        #self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        #self.layernorm_embedding = LayerNorm(args.embed_dim)
        self.model = TransformerEncoderBuilder.from_kwargs(
                n_layers=args.num_layers,
                n_heads=args.num_attention_heads,
                query_dimensions=args.embed_dim // args.num_attention_heads,
                value_dimensions=args.embed_dim // args.num_attention_heads,
                feed_forward_dimensions=4 * args.embed_dim,
                activation='gelu',
                #final_normalization=True,
                dropout=args.dropout,
                attention_type="causal-linear",
                #feature_map=Favor.factory(n_dims=self.d_model)
            ).get()
        #self.attn_mask = TriangularCausalMask(args.max_seq_len)
        self.lm_head = nn.Linear(
                args.embed_dim, len(task.target_dictionary), bias=False
            )
        self.apply(self._init_weights)
        # set zero embedding for padding symbol
        self.pad_idx = task.target_dictionary.pad()
        self.wte.weight.data[self.pad_idx].zero_()
        self.wpe.weight.data[0].zero_()

        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.embed_dim ** -0.5)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        prev_output_tokens,
        src_lengths = None
        # incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        # encoder_out=None,
    ):
        # print(prev_output_tokens.size())
        # print(prev_output_tokens)
        # if src_lengths is not None:
        #     print(src_lengths.size())
        #     print(src_lengths)

        features = self.extract_features(prev_output_tokens)#, incremental_state)
        lm_logits = self.lm_head(features)
        return (lm_logits,)

    def extract_features(
        self,
        prev_output_tokens
    ):

        bsz, seq_len = prev_output_tokens.size()
        attention_mask = prev_output_tokens.ne(self.pad_idx).long().to(prev_output_tokens.device)
        # set position ids to exclude padding symbols
        position_ids = attention_mask * (
            torch.arange(1, 1 + seq_len)
            .to(prev_output_tokens.device)
            .repeat(bsz, 1)
        )
        len_mask = LengthMask(torch.sum(attention_mask, axis=1), max_len=seq_len, device=prev_output_tokens.device)

        token_embeddings = self.wte(prev_output_tokens)
        position_embeddings = self.wpe(position_ids)
        x = self.drop(token_embeddings + position_embeddings)
        attn_mask = TriangularCausalMask(seq_len, device=x.device)
        outputs = self.model(x, attn_mask, len_mask)
        outputs = self.ln_f(outputs)

        return outputs

    def max_positions(self):
        return self.max_seq_len


@register_model_architecture("linear_transformer_lm", "linear_transformer_lm")
def base_architecture(args):
    if getattr(args, "max_seq_len", None) is None:
        args.max_seq_len = getattr(
            args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
        )
    args.embed_dim = getattr(args, "embed_dim", 768)
    args.num_attention_heads = getattr(args, "num_attention_heads", 12)
    args.num_layers = getattr(args, "num_layers", 12)
    args.dropout = getattr(args, "dropout", 0.1)

