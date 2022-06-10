import time, os, json
from collections import Counter
from pprint import pprint
from tqdm import tqdm
import subprocess#, multiprocessing
from functools import partial
from p_tqdm import p_uimap
RATIO = 4
MERGE_CNT = 700
CHAR_CNT = 128
WORKERS = 32

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encoding import pit2str, str2pit, ispitch

def resort(voc: str) -> str:
    assert(len(voc) % 2 == 0), voc
    pitch_set = list(set(voc[i:i+2] for i in range(0, len(voc), 2)))
    assert len(pitch_set) * 2 == len(voc), voc
    return ''.join(sorted(pitch_set, key=str2pit))

def gettokens(voc: set, merges):
    assert len(voc) > 1, voc
    last_idx = 0
    while(len(voc) > 1):
        flag = False
        for i in range(last_idx, len(merges)):
            t1, t2, t3 = merges[i]
            if t1 in voc and t2 in voc:
                voc.remove(t1)
                voc.remove(t2)
                voc.add(t3)
                flag = True
                last_idx = i+1
                break
        if not flag:
            break
    return voc

def merge_mulpies(new_toks, mulpies, other, merges, merged_vocs, divide_res):
    assert other is not None, mulpies
    for dur, mulpi in mulpies.items():
        if len(mulpi) > 1:  # apply bpe (with saved tokenization method)
            mulpi_sorted = tuple(sorted(list(mulpi), key=str2pit))
            if mulpi_sorted in divide_res:
                submulpies = divide_res[mulpi_sorted]
            else:
                submulpies = sorted(gettokens(set(str2pit(x) for x in mulpi_sorted), merges))

            for submulpi_num in submulpies:
                new_toks.extend([merged_vocs[submulpi_num], dur]+other)
        else:
            new_toks.extend([list(mulpi)[0], dur]+other)

def apply_bpe_for_sentence(toks, merges, merged_vocs, divide_res, ratio=RATIO):
    if isinstance(toks, str):
        toks = toks.split()
    new_toks = []
    mulpies = dict()
    other = None

    for idx in range(0, len(toks), ratio):
        e, d = toks[idx:idx+2]
        if not ispitch(e):
            if len(mulpies) > 0:
                merge_mulpies(new_toks, mulpies, other, merges, merged_vocs, divide_res)
                mulpies = dict()
            new_toks.extend(toks[idx:idx+ratio])
        else:
            mulpies.setdefault(d, set()).add(e) 
            other = toks[idx+2:idx+ratio]
    
    if len(mulpies) > 0:
        merge_mulpies(new_toks, mulpies, other, merges, merged_vocs, divide_res)

    assert len(new_toks) % ratio == 0, f'error new token len {len(new_toks)}'
 
    return new_toks

def load_before_apply_bpe(bpe_res_dir):
    merged_vocs = [pit2str(i) for i in range(CHAR_CNT)]
    merged_voc_to_int = {pit2str(i):i for i in range(CHAR_CNT)} 
    merges = []
    with open(bpe_res_dir+'codes.txt', 'r') as f:
        for line in f:
            a, b, _ = line.strip().split()
            a,b,ab = resort(a), resort(b), resort(a+b)
            
            a_ind, b_ind, ab_ind = merged_voc_to_int[a], merged_voc_to_int[b], len(merged_vocs)
            merges.append((a_ind, b_ind, ab_ind))

            merged_voc_to_int[ab] = ab_ind
            merged_vocs.append(ab)
    
    return merges, merged_vocs

def apply_bpe_for_word_dict(mulpi_list, merges):
    # apply bpe for vocabs
    bpe_freq = Counter()
    divided_bpe_total = Counter()
    divide_res = dict()
    for ori_voc, cnt in tqdm(mulpi_list):
        ret = sorted(gettokens(set(str2pit(x) for x in ori_voc), merges))
        divide_res[ori_voc] = ret
        divided_bpe_total[len(ret)] += cnt
        for r in ret:
            bpe_freq[merged_vocs[r]] += cnt

    return divide_res, divided_bpe_total, bpe_freq

def count_single_mulpies(toks, ratio=RATIO):
    if isinstance(toks, str):
        toks = toks.split()
    mulpies = dict()
    chord_dict = Counter()
    l_toks = len(toks)
    for idx in range(0, l_toks, ratio):
        e, d = toks[idx:idx+2]

        if not ispitch(e):
            if len(mulpies) > 0:
                for dur, mulpi in mulpies.items():
                    if len(mulpi) > 1:
                        chord_dict[tuple(sorted(list(mulpi), key=str2pit))] += 1
                mulpies = dict()
        else:
            mulpies.setdefault(d, set()).add(e)

    if len(mulpies) > 0:
        for dur, mulpi in mulpies.items():
            if len(mulpi) > 1:
                chord_dict[tuple(sorted(list(mulpi), key=str2pit))] += 1

    return chord_dict, l_toks // ratio


if __name__ == '__main__':
    start_time = time.time()

    paragraphs = []

    raw_data_path = 'data/preprocessed/raw_corpus.txt'
    merged_data_path = 'data/preprocessed/raw_corpus_bpe.txt'
    output_dir = 'data/bpe_res/'
    os.makedirs(output_dir, exist_ok=True)
    raw_data = []
    with open(raw_data_path, 'r') as f:
        for line in tqdm(f, desc="reading original txt file..."):
            raw_data.append(line.strip())

    chord_dict = Counter()
    before_total_tokens = 0
    for sub_chord_dict, l_toks in p_uimap(count_single_mulpies, raw_data, num_cpus=WORKERS):
        chord_dict += sub_chord_dict
        before_total_tokens += l_toks
    
    mulpi_list = sorted(chord_dict.most_common(), key=lambda x: (-x[1], x[0]))
    with open(output_dir+'ori_voc_cnt.txt', 'w') as f:
        f.write(str(len(mulpi_list)) + '\n')
        for k, v in mulpi_list:
            f.write(''.join(k) + ' ' + str(v) + '\n')
    with open(output_dir+'codes.txt', 'w') as stdout:
        with open(output_dir+'merged_voc_list.txt', 'w') as stderr:
            subprocess.run(['./music_bpe_exec', 'learnbpe', f'{MERGE_CNT}', output_dir+'ori_voc_cnt.txt'], stdout=stdout, stderr=stderr)
    print(f'learnBPE finished, time elapsed:　{time.time() - start_time}')
    start_time = time.time()

    merges, merged_vocs = load_before_apply_bpe(output_dir)
    divide_res, divided_bpe_total, bpe_freq = apply_bpe_for_word_dict(mulpi_list, merges)
    with open(output_dir+'divide_res.json', 'w') as f:
        json.dump({' '.join(k):v for k, v in divide_res.items()}, f)
    with open(output_dir+'bpe_voc_cnt.txt', 'w') as f:
        for voc, cnt in bpe_freq.most_common():
            f.write(voc + ' ' + str(cnt) + '\n')
    ave_len_bpe = sum(k*v for k, v in divided_bpe_total.items()) / sum(divided_bpe_total.values())
    ave_len_ori = sum(len(k)*v for k, v in mulpi_list) / sum(v for k, v in mulpi_list)
    print(f'average mulpi length original:　{ave_len_ori}, average mulpi length after bpe: {ave_len_bpe}')
    print(f'applyBPE for word finished, time elapsed:　{time.time() - start_time}')
    start_time = time.time()

    # applyBPE for corpus

    after_total_tokens = 0
    with open(merged_data_path, 'w') as f:
        for x in tqdm(raw_data, desc="writing bpe data"): # unable to parallelize for out of memory
            new_toks = apply_bpe_for_sentence(x, merges, merged_vocs, divide_res)
            after_total_tokens += len(new_toks) // RATIO
            f.write(' '.join(new_toks) + '\n')
    print(f'applyBPE for corpus finished, time elapsed:　{time.time() - start_time}')
    print(f'before tokens: {before_total_tokens}, after tokens: {after_total_tokens}, delta: {(before_total_tokens - after_total_tokens) / before_total_tokens}')