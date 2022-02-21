from fast_transformers.builders import TransformerEncoderBuilder, RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask, LengthMask

import logging, math
import os
import sys

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq import utils, metrics
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig
from fairseq.tasks import register_task
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.data import (
    MonolingualDataset, 
    TokenBlockDataset,
    plasma_utils,
    data_utils,
)
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion



logger = logging.getLogger(__name__)


DEFAULT_MAX_TARGET_POSITIONS = 1024
# INF = 2147483647

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
        assert not self.sentence_avg
        #TODO: adjust weight of evt losses and other losses by length (current strategy: simple average the losses)
        # weights = [sample["ntokens"]] + [sample["ontokens"]] * (len(losses) - 1)
        loss = torch.mean(torch.stack(losses))
        logging_output = {
            "loss": loss.data,
            "evt_loss": losses[0].data,
            "dur_loss": losses[1].data,
            "trk_loss": losses[2].data,
            "ins_loss": losses[3].data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample["ntokens"],
            "on_sample_size": sample["ntokens"],
        }
        return loss, sample["ntokens"], logging_output

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
        on_sample_size = sum(log.get("on_sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        # total_losses = 4
        # weighted_size = (sample_size + on_sample_size*(total_losses-1)) / total_losses
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "evt_loss", loss_evt / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "dur_loss", loss_dur / on_sample_size / math.log(2), on_sample_size, round=3
        )
        metrics.log_scalar(
            "trk_loss", loss_trk / on_sample_size / math.log(2), on_sample_size, round=3
        )
        metrics.log_scalar(
            "ins_loss", loss_ins / on_sample_size / math.log(2), on_sample_size, round=3
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

        # parser.add_argument('--max-pos-len', type=int, metavar='N', 
        #                     help='max positions in transformer')

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
        self.max_pos = args.tokens_per_sample
        #self.ratio = args.ratio
        #print("max positions:", self.max_pos)

        self.perm_inv = args.perm_inv
        if self.perm_inv > 1:
            self.wRpe = nn.Embedding(args.max_rel_pos+1, args.embed_dim) 
            self.wMpe = nn.Embedding(args.max_mea_pos+1, args.embed_dim)
        else:
            self.wpe = nn.Embedding(self.max_pos+1, args.embed_dim) # max_pos_len = 4096
        self.drop = nn.Dropout(args.dropout)
        self.ln_f = nn.LayerNorm(args.embed_dim, eps=1e-6)
        

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

        self.attn_mask = TriangularCausalMask(self.max_pos)
        self.proj_evt = nn.Linear(args.embed_dim, args.evt_voc_size, bias=False)
        self.proj_dur = nn.Linear(args.embed_dim, args.dur_voc_size, bias=False)
        self.proj_trk = nn.Linear(args.embed_dim, args.trk_voc_size, bias=False)
        self.proj_ins = nn.Linear(args.embed_dim, args.ins_voc_size, bias=False)

        self.apply(self._init_weights)
        # set zero embedding for padding symbol
        #TODO: check will the pad id be trained? (as TZ RZ YZ)
        self.pad_idx = task.target_dictionary.pad()
        self.wEvte.weight.data[self.pad_idx].zero_()
        self.wDure.weight.data[self.pad_idx].zero_()
        self.wTrke.weight.data[self.pad_idx].zero_()
        if self.perm_inv > 1:
            self.wRpe.weight.data[0].zero_()
            self.wMpe.weight.data[0].zero_()
        else:
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
        x,
        src_lengths=None,
    ):
        features = self.extract_features(x, src_lengths)
        evt_logits = self.proj_evt(features)
        dur_logits = self.proj_dur(features)
        trk_logits = self.proj_trk(features)
        ins_logits = self.proj_ins(features)

        return (evt_logits, dur_logits, trk_logits, ins_logits)

    def extract_features(
        self,
        x,
        src_lengths = None
    ):

        bsz, seq_len, ratio = x.size()
        evt_emb = self.wEvte(x[..., 0])

        # if not mapping to pad, padding idx will only occer at last
        evton_mask = x[..., 1].ne(self.pad_idx).float()[..., None].to(x.device) 

        tmp = self.wDure(x[..., 1])
        dur_emb = tmp * evton_mask
        # assert ((tmp==dur_emb).all())
        tmp = self.wTrke(x[..., 2])
        trk_emb = tmp * evton_mask
        # assert ((tmp==trk_emb).all())

        pad_mask = x[..., 0].ne(self.pad_idx).long().to(x.device)
        if src_lengths is not None:
            len_mask = LengthMask(src_lengths, max_len=seq_len, device=x.device)
        else:
            len_mask = LengthMask(torch.sum(pad_mask, axis=1), max_len=seq_len, device=x.device)
        

        if self.perm_inv > 1:
            rel_pos = pad_mask * x[..., 4]
            rel_pos_mask = rel_pos.ne(0).float()[..., None].to(x.device) # ignore bom, chord, eos

            measure_ids = pad_mask * x[..., 5]
            mea_mask = measure_ids.ne(0).float()[..., None].to(x.device) # ignore eos
            
            pos_emb = rel_pos_mask * self.wRpe(rel_pos) + mea_mask * self.wMpe(measure_ids)

        else:
            # set position ids to exclude padding symbols
            position_ids = pad_mask * (
                torch.arange(1, 1 + seq_len)
                .to(x.device)
                .repeat(bsz, 1)
            )
            pos_emb = self.wpe(position_ids)
        
        x = self.drop(evt_emb+dur_emb+trk_emb+pos_emb)


        outputs = self.model(x, self.attn_mask, len_mask)
        outputs = self.ln_f(outputs)
        
        return outputs

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
        return None



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
    def __init__(
        self,
        dataset,
        sizes,
        block_size,
        pad,
        eos,
        break_mode=None,
        include_targets=False,
        document_sep_len=1,
        ratio=4+1,
        sample_overlap_rate=4,
        permutation_invariant=3,
        trk_idx=2, # evt dur trk ins rel_pos mea_id
        spec_tok_cnt=4, # <bos> <pad> <eos> <unk>
        evt_vocab_size=425,
        trk_vocab_size=44,
    ):
        try:
            from fairseq.data.token_block_utils_fast import (
                _get_slice_indices_fast,
                _get_block_to_dataset_index_fast,
            )
        except ImportError:
            raise ImportError(
                "Please build Cython components with: `pip install --editable .` "
                "or `python setup.py build_ext --inplace`"
            )

        super(TokenBlockDataset, self).__init__()
        self.dataset = dataset
        self.pad = pad
        self.eos = eos
        self.include_targets = include_targets


        self.ratio = ratio
        self.perm_inv = permutation_invariant
        self.sample_len_max = block_size
        
        self.trk_idx = trk_idx
        self.cc_idx = evt_vocab_size - 1
        self.spec_tok_cnt = spec_tok_cnt
        self.max_trk_cnt = trk_vocab_size - spec_tok_cnt

        assert len(dataset) == len(sizes)
        assert len(dataset) > 0

        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=np.int64)
        else:
            if torch.is_tensor(sizes):
                sizes = sizes.numpy()
            sizes = sizes.astype(np.int64)

        break_mode = break_mode if break_mode is not None else "complete_doc"
        assert break_mode == 'complete_doc', break_mode



        sizes_cs = np.cumsum(sizes)
        piece_sep_ids = np.where(sizes == document_sep_len)[0].tolist()
        totpieces = len(piece_sep_ids)
        slice_indices = np.zeros((totpieces,2), dtype=int)
        block_to_dataset_index = np.zeros((totpieces,3), dtype=int)

        for i in range(len(piece_sep_ids)):
            s = piece_sep_ids[i-1] if i > 0 else -1
            e = piece_sep_ids[i]
            slice_indices[i, :] = (sizes_cs[s] if s >= 0 else 0, sizes_cs[e-1])
            block_to_dataset_index[i, :] = (s+1, 0, e-1)

        
        # slice_indices_std = _get_slice_indices_fast(
        #     sizes, str(break_mode), INF, document_sep_len
        # )
        # assert((slice_indices == slice_indices_std).all())
        # block_to_dataset_index_std = _get_block_to_dataset_index_fast(
        #     sizes,
        #     slice_indices,
        # )
        # assert((block_to_dataset_index == block_to_dataset_index_std).all())
 

        #print(slice_indices.shape)
        sample_step = max(round(self.sample_len_max / sample_overlap_rate), 1) 
        new_slice_indices = []
        new_block_to_dataset_index = []
        for line, line_piece in zip(slice_indices, block_to_dataset_index):
            l_piece_tot = line[1] - line[0]
            assert l_piece_tot % self.ratio == 0, (line[0], line[1])
            l_toks = l_piece_tot // self.ratio
            chosen_cnt = math.ceil((l_toks + np.random.randint(sample_step)) / sample_step)
            #chosen_cnt = sum(1 for _ in range(0 - np.random.randint(sample_step), l_toks, sample_step))
            new_slice_indices.append(np.stack([line]*chosen_cnt))
            new_block_to_dataset_index.append(np.stack([line_piece]*chosen_cnt))

        slice_indices = np.concatenate(new_slice_indices)
        block_to_dataset_index = np.concatenate(new_block_to_dataset_index)
        #print(slice_indices.shape)

        self._sizes = slice_indices[:, 1] - slice_indices[:, 0]
        self._sizes[:] = self.sample_len_max

        self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
        self._sizes = plasma_utils.PlasmaArray(self._sizes)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(block_to_dataset_index)

    def __getitem__(self, index):
        # start_ds_idx means measure number
        # start_offset must be 0
        # end_ds_idx means after {sample_len_max} tokens, which measure the end token in
       
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
        assert start_offset == 0, (start_ds_idx, start_offset, end_ds_idx)
        
        st = np.random.randint(start_ds_idx, end_ds_idx+1)

        #print(start_ds_idx, end_ds_idx)
        buffer = []
        cur_len = 0
        for idx in range(st, end_ds_idx+1):
            tmp = self.dataset[idx].view(-1, self.ratio)
            if self.perm_inv % 2 == 1: # swap cc, pos(data aug for auto-regressive)
                #TODO: swap pos
                all_cc_pos = torch.nonzero(tmp[..., 0] == self.cc_idx).view(-1).tolist() # find all cc indexs
                all_cc_pos.append(tmp.size(0))
                to_swap = []
                for pos, nexp in zip(all_cc_pos[:-1], all_cc_pos[1:]): # split to list
                    to_swap.append(tmp[pos:nexp, ...])
                # to_swap_idx = list(range(len(to_swap)))
                # random.shuffle(to_swap_idx)
                to_swap_idx = torch.randperm(len(to_swap))
                tmp = torch.cat([tmp[:all_cc_pos[0], ...]] + [to_swap[x] for x in to_swap_idx])
                #assert not (tmp == self.dataset[idx].view(-1, self.ratio)).all(), (to_swap, all_cc_pos)
            mea = (idx-st+1) * 3
            # mea_list = [[mea-2], [mea-1]] + [[mea]]*(tmp.size(0)-2)
            mea_num = torch.zeros((tmp.size(0),1), dtype=int)
            mea_num[2:, 0] = mea
            mea_num[1][0] = mea-1
            mea_num[0][0] = mea-2
            buffer.append(torch.cat((tmp, mea_num), dim=1))
            cur_len += tmp.size(0)
            if cur_len >= self.sample_len_max:
                break
        

        buffer = torch.cat(buffer)
        if cur_len < self.sample_len_max:
            buffer = torch.cat([buffer, buffer.new([[self.eos]*(self.ratio+1)])])
   

        item = buffer[:self.sample_len_max, ...]
        if self.perm_inv > 0:
            #TODO: should we assure drum track always be track 0? (give model some info)
            perm = torch.cat([torch.arange(self.spec_tok_cnt), torch.randperm(self.max_trk_cnt) + self.spec_tok_cnt])
            item[..., self.trk_idx].apply_(lambda x: perm[x])
            # cmp = self.dataset[st].view(-1, self.ratio)[..., self.trk_idx]
            # assert not (item[:cmp.size(0), self.trk_idx] == cmp).all()

        assert self.include_targets

        # *target* is the original sentence (=item)
        # *source* is shifted right by 1 (maybe left-padded with eos)
        # *past_target* is shifted right by 2 (left-padded as needed)
        # <eos> rel_pos is 0, mea_id is 0
        source = torch.cat([item.new([[self.eos]*(self.ratio-1) + [0, 0]]), item[:-1, ...]])
        on = torch.sum(item[:, 1].ne(self.pad)).item() # if no mapping to pad, on will be item.size(0)
        #print(item.size(), on)
        # past_target = torch.cat(
        #     [item.new([[self.pad]*(self.ratio+1), [self.eos]*(self.ratio+1)]), item[:-2, ...]]
        # )
        
        return source, item, on

def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values) # max batch size
 
    res = values[0].new(len(values), size, values[0].size(-1)).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])

    return res

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
                    collate_tokens(
                        [s[key][i] for s in samples],
                        pad_idx,
                        eos_idx,
                        left_pad=False,
                    )
                )
            return res
        else:
            return collate_tokens(
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

    #print(torch.LongTensor([s["source"].numel() // ratio for s in samples]))
    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(s["source"].size(0)  for s in samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": torch.LongTensor([s["source"].size(0) for s in samples]),
        },
        "target": target,
        "ontokens": sum(s["on"] for s in samples)
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
        assert not self.add_bos_token, "<bos> is occupied"

        assert targets is None or all(
            t in {"self", "future", "past"} for t in targets
        ), "targets must be none or one of 'self', 'future', 'past'"
        if targets is not None and len(targets) == 0:
            targets = None
        assert len(targets) == 1 and targets[0] == 'future'
        self.targets = targets
    def collater(self, samples):
        return collate(samples, self.vocab.pad(), self.vocab.eos())

    def __getitem__(self, index):
        assert self.targets is not None
        source, target, on = self.dataset[index]
        source, target = self._make_source_target(
                source, target, None
            )

        source, target = self._maybe_add_bos(source, target)
        return {"id": index, "source": source, "target": target, "on": on}



@dataclass
class SymphonyModelingConfig(LanguageModelingConfig):
    
    ratio: int = field(
        default=4, metadata={"help": "note/metadata attribute amount: default (evt, dur, trk, ins)"}
    )
    evt_voc_size: int = field(
        default=-1, metadata={"help": "event vocab size"}
    )
    dur_voc_size: int = field(
        default=-1, metadata={"help": "duration vocab size"}
    )
    trk_voc_size: int = field(
        default=-1, metadata={"help": "track vocab size"}
    )
    ins_voc_size: int = field(
        default=-1, metadata={"help": "instrument vocab size"}
    )
    max_rel_pos: int = field(
        default=-1, metadata={"help": "maximum relative position index, calculated by make_data.py"}
    )
    max_mea_pos: int = field(
        default=-1, metadata={"help": "maximum measure cnt within a sample, calculated by make_data.py"}
    )
    perm_inv: int = field(
        default=3, metadata={"help": "consider permutation invariance for music, 0: without PI, 1: data augmentation only, 2: positional encoding only, 3: all considered"}
    )
    sample_overlap_rate: int = field(
        default=4, metadata={"help": "sample overlap rate, default is 4 (stride 1024), also needed in make_data.py"}
    )

@register_task("symphony_modeling", dataclass=SymphonyModelingConfig)
class SymphonyModelingTask(LanguageModelingTask):
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )
        #print('load indexed dataset finished')
        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )
        #print('maybe_shorten_dataset finished')
        dataset = TupleMultiHeadDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
            ratio=self.args.ratio + 1,
            sample_overlap_rate=self.args.sample_overlap_rate,
            permutation_invariant=self.args.perm_inv,
            #trk_idx=self.args.trk_idx,
            #spec_tok_cnt=self.args.spec_tok_cnt,
            evt_vocab_size=self.args.evt_voc_size,
            trk_vocab_size=self.args.trk_voc_size,
        )
        #print('TupleMultiHeadDataset init finished')
        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )

        self.datasets[split] = self._initialize_dataset(
            dataset=dataset,
            sizes=dataset.sizes,
            src_vocab=self.dictionary,
            tgt_vocab=self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
        )
        #print('_initialize_dataset finished')

    def _initialize_dataset(self, **kwargs):
        return MultiheadDataset(**kwargs)
    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        assert False, "inference not implemented"
# fairseq.tasks.language_modeling.TokenBlockDataset = TupleMultiHeadDataset
# fairseq.tasks.language_modeling.MonolingualDataset = MultiheadDataset
