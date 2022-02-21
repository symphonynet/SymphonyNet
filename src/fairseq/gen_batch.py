import os, time

from more_itertools.more import last

import json
import numpy as np
import torch
import copy
from tqdm import tqdm
from miditoolkit.midi.containers import Note as mtkNote
from miditoolkit.midi.parser import MidiFile
from miditoolkit.midi.containers import Instrument
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encoding import ison, char2int, str2pit, ispitch
from preprocess.preprocess_midi import midi_to_event_seq_str
from preprocess.get_bpe_data import apply_bpe_for_sentence, load_before_apply_bpe
from make_data import process_single_piece

def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs

def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word

def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def sampling(logit, p=None, t=1.0):
    #print(logit.shape)
    logit = logit.squeeze().cpu().numpy()
    
    probs = softmax_with_temperature(logits=logit, temperature=t)
    #print(probs)
    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


PAD = 1
EOS = 2
BOS = 0

RATIO = 4
NEW_VER = 1
MAX_POS_LEN = 4096 if NEW_VER == 1 else 4097
PI_LEVEL = 2
IGNORE_META_LOSS = 1
NOTON_PAD = BOS if IGNORE_META_LOSS == 0 else PAD
NOTON_PAD_DUR = NOTON_PAD if NEW_VER == 1 else 36
NOTON_PAD_TRK = NOTON_PAD if NEW_VER == 1 else 44
BPE = "_bpe"
# BPE = ""
DATA_BIN=f"linear_{MAX_POS_LEN}_chord{BPE}_hardloss{IGNORE_META_LOSS}"
CHECKPOINT_SUFFIX=f"{DATA_BIN}_PI{PI_LEVEL}"
DATA_BIN_DIR=f"../../data/model_spec/{DATA_BIN}/bin/"
DATA_VOC_DIR=f"../../data/model_spec/{DATA_BIN}/vocabs/"


from fairseq.models import FairseqLanguageModel
custom_lm = FairseqLanguageModel.from_pretrained('.', 
    checkpoint_file=f'../../ckpt/checkpoint_last_{CHECKPOINT_SUFFIX}.pt', 
    data_name_or_path=DATA_BIN_DIR, 
    user_dir="linear_transformer_inference" + ("" if NEW_VER == 1 else "_old"))
print(CHECKPOINT_SUFFIX)

vocabs = {}
voc2int = {}


def index2word(typ, i):
    if NEW_VER == 0 and typ != 0 and str(i) not in vocabs[typ]:
        i = i % len(vocabs[typ])
    return vocabs[typ][str(i)]
def word2index(typ, i):
    return voc2int[typ][i]
for i in range(RATIO):
    with open(f'{DATA_VOC_DIR}vocab_{i}.json', 'r') as f:
        vocabs[i] = json.load(f)
        voc2int[i] = {v:int(k)for k, v in vocabs[i].items()}


with open(f'{DATA_VOC_DIR}ori_dict.json', 'r') as f:
    str2int = json.load(f)
if NEW_VER == 1:
    str2int.update({x:(PAD if IGNORE_META_LOSS == 1 else BOS) for x in ('RZ', 'TZ', 'YZ')}) 

# for BPE
if BPE != "":
    merges, merged_vocs = load_before_apply_bpe("../preprocess/bpe_res/")
else:
    merges, merged_vocs = None, None

prime_chords = None
prime_mea_idx = 0

def process_prime_midi(prime_midi_path, max_measures, max_chord_measures, perm_inv = PI_LEVEL, ratio=RATIO, sample_len_max=MAX_POS_LEN, bpe=0 if BPE == "" else 1):
    
    toks, _ = midi_to_event_seq_str(prime_midi_path, readonly=True)
    #print(_)
    if bpe > 0:
        assert merges is not None
        toks = apply_bpe_for_sentence(toks, merges, merged_vocs, {})
        
    measures, _, _, _ = process_single_piece((toks, str2int), ratio, sample_len_max)

    prime_nums = [[EOS]*ratio + [0, 0]]
    prime_nums[0][3] = 1 # set instrument to vanilla pos
    ins_label = [EOS]

    
    trk_map = np.concatenate([np.arange(4), np.random.permutation(40) + 4])

    global prime_chords
    prime_chords = [index2word(0, x[ratio+1]) for x in measures[:max_chord_measures]]
    for mea_id, mea in enumerate(measures):
        if mea_id >= max_measures:
            break
        assert len(mea) % (ratio+1) == 0, mea
        print(index2word(0, mea[0]), index2word(0, mea[ratio+1]))
        
        if perm_inv % 2 == 1:
            auged_measure = []
            cc_list = []
            cur_cc = []
            for id in range(0, len(mea), ratio+1):
                cur_tok = mea[id:id+ratio+1]
                if id <= ratio + 1:
                    auged_measure += cur_tok
                    continue
    
                if cur_tok[0] == str2int['NT'] and len(cur_cc) > 0:
                    cc_list.append(cur_cc)
                    cur_cc = []
                cur_cc += cur_tok
            if len(cur_cc) > 0:
                cc_list.append(cur_cc)
                cur_cc = []

            if len(cc_list) > 1:
                new_order = np.random.permutation(len(cc_list))
                for i in new_order:
                    auged_measure += cc_list[i]
            else:
                for cc in cc_list:
                    auged_measure += cc

            assert len(auged_measure) == len(mea), (len(auged_measure), len(mea))
            mea = auged_measure

        for id in range(0, len(mea), ratio+1):
            mea_pos = (mea_id+1) * 3
            if id == 0:
                mea_pos -= 2
            elif id == ratio+1:
                mea_pos -= 1
            cur_tok = mea[id:id+ratio+1] + [mea_pos]
            ins_label.append(cur_tok[3])
            cur_tok[3] = len(prime_nums) + 1
            if perm_inv > 0:
                cur_tok[2] = trk_map[cur_tok[2]]
            prime_nums.append(cur_tok)

    return prime_nums, ins_label

m = custom_lm.models[0]
m.cuda()
m.eval()

def get_next_chord(ori):
    global prime_chords
    assert prime_chords is not None
    global prime_mea_idx
    if prime_mea_idx < len(prime_chords):
        ret = prime_chords[prime_mea_idx]
        prime_mea_idx += 1
        return ret
    else:
        return ori

def get_next(p, memory, has_prime = False):
    pr = torch.from_numpy(np.array(p))[None, None, :].cuda()

    (e,d,t,ins), memory = m(src_tokens=pr, src_lengths=memory)
    e, d, t, ins = e[0,:], d[0,:], t[0,:], ins[0,:]
    if has_prime:
        return (np.int64(EOS), np.int64(EOS), np.int64(EOS), ins), memory
    evt =  sampling(e)
    while evt == EOS:
        return (evt, np.int64(EOS), np.int64(EOS), ins), memory
        # evt =  sampling(e)
    evt_word = index2word(0, evt)
    if evt_word.startswith('H'):
        rep = get_next_chord(evt_word)
        print(evt_word, rep)
        return (np.int64(word2index(0, rep)), np.int64(NOTON_PAD_DUR), np.int64(NOTON_PAD_TRK), ins), memory
    if not ison(evt_word):
        return (evt, np.int64(NOTON_PAD_DUR), np.int64(NOTON_PAD_TRK), ins), memory

    dur = sampling(d)
    while dur == EOS:
        dur = sampling(d)
    while dur == NOTON_PAD_DUR:
        dur = sampling(d)
       
    trk = sampling(t, p=0)
    

    return (evt, dur, trk, ins), memory
def is_bom(idx):
    return index2word(0, idx)[0].lower() == 'm'


def calc_pos(evt_tok, last_rel_pos, last_mea_pos):
    assert evt_tok != EOS, 'no eos pos'
    typ = index2word(0, evt_tok)[0].lower()
    if typ == 'm':
        if (last_mea_pos+1) % 3 == 0: #empty measure
            last_mea_pos += 1
        assert (last_mea_pos+1) % 3 == 1, f'error <bos> measure pos {last_mea_pos+1}' #TODO: empty measure
        return 0, last_mea_pos + 1
    elif typ == 'h':
        assert (last_mea_pos+1) % 3 == 2, f'there must be a <bom> before a chord {last_mea_pos+1}'
        return 0, last_mea_pos + 1
    elif typ == 'n':
        if last_mea_pos % 3 == 2:
            last_mea_pos += 1
        assert last_mea_pos % 3 == 0, f'mea pos of <cc> must be a multiple of 3 {last_mea_pos}'
        return 1, last_mea_pos
    elif typ == 'p':
        assert last_mea_pos % 3 == 0, f'mea pos of <pos> must be a multiple of 3 {last_mea_pos}'
        assert (last_rel_pos+1) % 2 == 0, f'rel pos of <pos> must be even {last_rel_pos+1}'
        return last_rel_pos+1, last_mea_pos
    
    assert last_mea_pos % 3 == 0, f'mea pos of <on> must be a multiple of 3 {last_mea_pos}'
    if last_rel_pos % 2 == 0: # last token is a <pos>
        last_rel_pos += 1
    
    return last_rel_pos, last_mea_pos # on

def gen_one(prime_nums):
    global prime_mea_idx
    prime_mea_idx = 0
    prime = copy.deepcopy(prime_nums)
    ins_list = [-1] 
    MAX_LEN = 4090
    with torch.no_grad():
        memo = None
        cur_rel_pos = 0
        cur_mea = 0
        for item, next_item in zip(prime[:-1], prime[1:]):
            #print(item)
            (e,d,t,ins), memo = get_next(item, memo, has_prime=True)
            cur_rel_pos, cur_mea = calc_pos(next_item[0], cur_rel_pos, cur_mea)
            ins_list.append(ins)
        print(len(prime), len(ins_list))

        (e,d,t,ins), memo = get_next(prime[-1], memo, has_prime=False)
        cur_rel_pos, cur_mea = calc_pos(e, cur_rel_pos, cur_mea)
        #assert isinstance(t, np.int64)
        prime.append((e,d,t,len(prime)+1, cur_rel_pos, cur_mea))
        ins_list.append(ins)

        for i in tqdm(range(MAX_LEN - len(prime))):
            (e,d,t,ins), memo = get_next(prime[-1], memo)
            if t == EOS:
                assert len(prime) > MAX_LEN // 4
                break
            cur_rel_pos, cur_mea = calc_pos(e, cur_rel_pos, cur_mea)
            #assert isinstance(t, np.int64)
            prime.append((e,d,t,len(prime)+1, cur_rel_pos, cur_mea))
            ins_list.append(ins)

    return prime, ins_list

def get_trk_ins_map(prime, ins_list):
    track_map = {}
    print(len(prime), len(ins_list))
    idx = 0
    for (e,d,t,_, _, _),ins in zip(prime, ins_list):
        ee = index2word(0, e)
        # dd = index2word(1, d)
        # tt = index2word(2, t)
        idx += 1

        if ison(ee):
            if t == NOTON_PAD_TRK:
                print(e,d,t, idx)
            track_map.setdefault(t, []).append(ins)
    trk_ins_map = {}
    for k in track_map:
        v = torch.stack(track_map[k])
        print(k, v.size())
        logits = torch.mean(v, axis=0)
        #print(logits)
        ins_word = sampling(logits,p=0.9)
        trk_ins_map[k] = ins_word
    print(trk_ins_map)
    return trk_ins_map

def get_note_seq(prime, trk_ins_map):
    note_seq = []
    # print(prime[0])
    measure_time = 0
    last_bom = 0
    error_note = 0
    for (e,d,t,_, _, _) in prime[1:]:

        ee = index2word(0, e)

        #print(ee,dd,tt,ii)
        if ee[0].lower() == 'm':
            # evt = 'BOM'
            #print(ee)
            measure_time += last_bom
            last_bom = char2int(ee[1])+(62 if ee[0] == 'M' else 0)
            last_pos = -1
        elif ee[0].lower() == 'p':
            # evt = 'POS'
            last_pos = char2int(ee[1]) + (62 if ee[0] == 'P' else 0)
        elif ee == 'NT':
            last_pos = -1
        elif ee[0].lower() == 'h':
            pass #print(ee)
        elif ison(ee):
            if t != NOTON_PAD_TRK and d != NOTON_PAD_DUR:
                dd = index2word(1, d)
                tt = index2word(2, t)
                assert last_pos != -1, 'there must be a <pos> between <on> and <cc>'
                start = measure_time + last_pos
                trk = char2int(tt[1])+(62 if tt[0] == 'T' else 0)
                dur = char2int(dd[1])+(62 if dd[0] == 'R' else 0)

                for i in range(0, len(ee), 2):
                    eee = ee[i:i+2]
                    note_seq.append((str2pit(eee), trk_ins_map[t]-4, start, start + dur, trk))
            else:
                error_note += 1
        else:
            assert False, (ee, d, t)
    print(f'error note cnt: {error_note}')
    return note_seq


def note_seq_to_midi_file(note_seq, filename, ticks_per_beat=480):

    tickes_per_32th = ticks_per_beat // 8
    tracks = {}
    for pitch, program, start, end, track_id in note_seq:

        tracks.setdefault((track_id, program), []).append(mtkNote(90, pitch, start * tickes_per_32th, end * tickes_per_32th))

    midi_out = MidiFile(ticks_per_beat=ticks_per_beat)

    for tp, notes in tracks.items():
        program = tp[1]
        instrument = Instrument(program % 128, is_drum=program >= 128)
        instrument.notes = notes
        instrument.remove_invalid_notes(verbose=False)
        midi_out.instruments.append(instrument)
    midi_out.dump(filename)

GEN_DIR = f'../../generated/{CHECKPOINT_SUFFIX}/'
os.makedirs(GEN_DIR, exist_ok=True)
if __name__ == '__main__':
    midi_name = sys.argv[1]
    max_measure_cnt = int(sys.argv[2])
    max_chord_measure_cnt = int(sys.argv[3])
    prime, ins_label = process_prime_midi(f'{midi_name}.mid', max_measure_cnt, max_chord_measure_cnt)
    gen_cnt = int(sys.argv[4])
    for i in range(gen_cnt):
        while(True):
            try:
                generated, ins_logits = gen_one(prime)
                break
            except Exception as e:
                print(e)
                continue
        trk_ins_map = get_trk_ins_map(generated, ins_logits)
        note_seq = get_note_seq(generated, trk_ins_map)
        print(len(note_seq))
        #print(note_seq) 
        timestamp = time.strftime("%m-%d_%H-%M-%S", time.localtime()) 
        note_seq_to_midi_file(note_seq, f'{GEN_DIR}{midi_name}_prime{max_measure_cnt}_chord{max_chord_measure_cnt}_{timestamp}.mid')