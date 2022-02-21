from fast_transformers.builders import TransformerEncoderBuilder, RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask, LengthMask

import logging, math
import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq.tasks.language_modeling
from fairseq import utils, metrics
from fairseq.tasks import language_modeling, register_task
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.data import (
    MonolingualDataset, 
    TokenBlockDataset,
    data_utils
)
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion



logger = logging.getLogger(__name__)


DEFAULT_MAX_TARGET_POSITIONS = 1024

@register_criterion("multiple_loss")
class MultiplelossCriterion(CrossEntropyCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        losses = self.compute_loss(model, net_output, sample, reduce=reduce) # return a list
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        loss = torch.mean(torch.stack(losses))
        logging_output = {
            "loss": loss.data,
            "evt_loss": losses[0].data,
            "dur_loss": losses[1].data,
            "trk_loss": losses[2].data,
            "ins_loss": losses[3].data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs_tuple = model.get_normalized_probs(net_output, log_probs=True)
        losses = []
        for idx, lprobs in enumerate(lprobs_tuple):
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, net_output)[..., idx].view(-1)

            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )
            losses.append(loss)
        return losses

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_evt = sum(log.get("evt_loss", 0) for log in logging_outputs)
        loss_dur = sum(log.get("dur_loss", 0) for log in logging_outputs)
        loss_trk = sum(log.get("trk_loss", 0) for log in logging_outputs)
        loss_ins = sum(log.get("ins_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "evt_loss", loss_evt / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "dur_loss", loss_dur / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "trk_loss", loss_trk / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ins_loss", loss_ins / sample_size / math.log(2), sample_size, round=3
        )

        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
            metrics.log_derived(
                "evt_ppl", lambda meters: utils.get_perplexity(meters["evt_loss"].avg)
            )
            metrics.log_derived(
                "dur_ppl", lambda meters: utils.get_perplexity(meters["dur_loss"].avg)
            )
            metrics.log_derived(
                "trk_ppl", lambda meters: utils.get_perplexity(meters["trk_loss"].avg)
            )
            metrics.log_derived(
                "ins_ppl", lambda meters: utils.get_perplexity(meters["ins_loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

@register_model("linear_transformer_multi")
class LinearTransformerMultiHeadLM(FairseqLanguageModel):
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
        parser.add_argument('--evt-voc-size', type=int, metavar='N', 
                            help='event vocab size')
        parser.add_argument('--dur-voc-size', type=int, metavar='N', 
                            help='duration vocab size')
        parser.add_argument('--trk-voc-size', type=int, metavar='N', 
                            help='track vocab size')
        parser.add_argument('--ins-voc-size', type=int, metavar='N', 
                            help='instrument vocab size')
        parser.add_argument('--max-pos-len', type=int, metavar='N', 
                            help='max positions in transformer')
        parser.add_argument('--ratio', type=int, metavar='N', 
                            help='how many tokens on a position')
        # parser.add_argument('--attention-dropout', type=float, metavar='D',
        #                     help='dropout probability for attention weights')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        return cls(LinearTransformerMultiHeadDecoder(args, task))


class LinearTransformerMultiHeadDecoder(FairseqDecoder):
    def __init__(self, args, task):

        super().__init__(task.target_dictionary)
        #print(task.target_dictionary)
        # for i in range(len(task.target_dictionary)):
        #     print(i, task.target_dictionary[i])
        self.embed_dim = args.embed_dim
        self.wEvte = nn.Embedding(args.evt_voc_size, args.embed_dim)
        self.wTrke = nn.Embedding(args.trk_voc_size, args.embed_dim)
        self.wDure = nn.Embedding(args.dur_voc_size, args.embed_dim)
        self.max_pos = args.max_pos_len
        self.ratio = args.ratio
        #print("max positions:", self.max_pos)
        self.wpe = nn.Embedding(self.max_pos+1, args.embed_dim) # max_pos_len = 4097 , not 4097*4
        self.drop = nn.Dropout(args.dropout)
        self.ln_f = nn.LayerNorm(args.embed_dim, eps=1e-6)
        

        self.model = RecurrentEncoderBuilder.from_kwargs(
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
        self.proj_evt = nn.Linear(args.embed_dim, args.evt_voc_size, bias=False)
        self.proj_dur = nn.Linear(args.embed_dim, args.dur_voc_size, bias=False)
        self.proj_trk = nn.Linear(args.embed_dim, args.trk_voc_size, bias=False)
        self.proj_ins = nn.Linear(args.embed_dim, args.ins_voc_size, bias=False)

        self.apply(self._init_weights)
        # set zero embedding for padding symbol
        self.pad_idx = task.target_dictionary.pad()
        self.wEvte.weight.data[self.pad_idx].zero_()
        self.wDure.weight.data[self.pad_idx].zero_()
        self.wTrke.weight.data[self.pad_idx].zero_()
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
        src_lengths=None,
    ):
        features, memory = self.extract_features(prev_output_tokens, src_lengths)
        evt_logits = self.proj_evt(features)
        dur_logits = self.proj_dur(features)
        trk_logits = self.proj_trk(features)
        ins_logits = self.proj_ins(features)

        return (evt_logits, dur_logits, trk_logits, ins_logits), memory

    def extract_features(
        self,
        prev_output_tokens,
        src_lengths = None
    ):

        bsz, seq_len, ratio = prev_output_tokens.size()
        #print(bsz, seq_len, ratio)
        evt_emb = self.wEvte(prev_output_tokens[..., 0])
        dur_emb = self.wDure(prev_output_tokens[..., 1])
        trk_emb = self.wTrke(prev_output_tokens[..., 2])
        pos_emb = self.wpe(prev_output_tokens[..., 3])
        
        x = self.drop(evt_emb+dur_emb+trk_emb+pos_emb)

        outputs, memory = self.model(x.squeeze(0), src_lengths)
        outputs = self.ln_f(outputs)
        
        return outputs, memory
    
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if log_probs:
            return tuple(utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace) for logits in net_output)
        else:
            return tuple(utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace) for logits in net_output)

    def max_positions(self):
        return self.max_pos * self.ratio



@register_model_architecture("linear_transformer_multi", "linear_transformer_multi")
def base_architecture(args):
    
    args.embed_dim = getattr(args, "embed_dim", 512)
    args.num_attention_heads = getattr(args, "num_attention_heads", 16)
    args.num_layers = getattr(args, "num_layers", 12)
    args.dropout = getattr(args, "dropout", 0.1)

@register_model_architecture("linear_transformer_multi", "linear_transformer_multi_large")
def base_architecture(args):
    args.embed_dim = getattr(args, "embed_dim", 768)
    args.num_attention_heads = getattr(args, "num_attention_heads", 12)
    args.num_layers = getattr(args, "num_layers", 12)
    args.dropout = getattr(args, "dropout", 0.1)


class TupleMultiHeadDataset(TokenBlockDataset):
    def __getitem__(self, index):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]

        buffer = torch.cat(
            [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
        )

        slice_s, slice_e = self.slice_indices[index]
        length = slice_e - slice_s

        s, e = start_offset, start_offset + length
        item = buffer[s:e]


        if self.include_targets:
            # *target* is the original sentence (=item)
            # *source* is shifted right by 1 (maybe left-padded with eos)
            # *past_target* is shifted right by 2 (left-padded as needed)
            ratio = 4 # event, duration, track, instrument
            if s == 0:
                source = torch.cat([item.new([self.eos] * ratio), buffer[0 : e - ratio]])
                past_target = torch.cat(
                    [item.new([self.pad, self.eos] * ratio), buffer[0 : e - ratio*2]]
                )
            else:
                source = buffer[s - ratio : e - ratio]
                if s == ratio:
                    past_target = torch.cat([item.new([self.eos]*ratio), buffer[0 : e - 2*ratio]])
                else:
                    past_target = buffer[s - 2*ratio: e - 2*ratio]

            return source, item, past_target

        return item

# pad = 1, eos = 2
def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}
    # print('raw length', end = ' ')
    # for s in samples:
    #     print(len(s['source']), end = ' ')
    # print()
    def merge(key, is_list=False):
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                res.append(
                    data_utils.collate_tokens(
                        [s[key][i] for s in samples],
                        pad_idx,
                        eos_idx,
                        left_pad=False,
                    )
                )
            return res
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
            )

    src_tokens = merge("source")
    if samples[0]["target"] is not None:
        is_target_list = isinstance(samples[0]["target"], list)
        target = merge("target", is_target_list)
    else:
        target = src_tokens
    ratio = 4
    #print(torch.LongTensor([s["source"].numel() // ratio for s in samples]))
    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(len(s["source"]) // ratio for s in samples),
        "net_input": {
            "src_tokens": src_tokens.view(src_tokens.size(0), -1, ratio),
            "src_lengths": torch.LongTensor([s["source"].numel() // ratio for s in samples]),
        },
        "target": target.view(target.size(0), -1, ratio)
    }

class MultiheadDataset(MonolingualDataset):
    def __init__(
        self,
        dataset,
        sizes,
        src_vocab,
        tgt_vocab,
        add_eos_for_other_targets,
        shuffle,
        targets=None,
        add_bos_token=False,
    ):
        # print(len(sizes))
        # print(type(dataset))
        # print(len(dataset))
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle
        self.add_bos_token = add_bos_token

        assert targets is None or all(
            t in {"self", "future", "past"} for t in targets
        ), "targets must be none or one of 'self', 'future', 'past'"
        if targets is not None and len(targets) == 0:
            targets = None
        self.targets = targets
    def collater(self, samples):
        return collate(samples, self.vocab.pad(), self.vocab.eos())

fairseq.tasks.language_modeling.TokenBlockDataset = TupleMultiHeadDataset
fairseq.tasks.language_modeling.MonolingualDataset = MultiheadDataset
