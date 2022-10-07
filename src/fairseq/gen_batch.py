import os, sys, time
import torch

MAX_POS_LEN = 4096
PI_LEVEL = 2
IGNORE_META_LOSS = 1

BPE = "_bpe"
# BPE = ""
DATA_BIN=f"linear_{MAX_POS_LEN}_chord{BPE}_hardloss{IGNORE_META_LOSS}"
CHECKPOINT_SUFFIX=f"{DATA_BIN}_PI{PI_LEVEL}"
DATA_BIN_DIR=f"data/model_spec/{DATA_BIN}/bin/"
DATA_VOC_DIR=f"data/model_spec/{DATA_BIN}/vocabs/"
from gen_utils import process_prime_midi, gen_one, get_trk_ins_map, get_note_seq, note_seq_to_midi_file, music_dict
music_dict.load_vocabs_bpe(DATA_VOC_DIR, 'data/bpe_res/' if BPE == '_bpe' else None)


from fairseq.models import FairseqLanguageModel
custom_lm = FairseqLanguageModel.from_pretrained('.',
    checkpoint_file=f'ckpt/checkpoint_last_{CHECKPOINT_SUFFIX}.pt',
    data_name_or_path=DATA_BIN_DIR,
    user_dir="src/fairseq/linear_transformer_inference")
print(f'Generation using model: {CHECKPOINT_SUFFIX}')

m = custom_lm.models[0]
# TODO: make this a flag of some sort?
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    m.cuda()
m.eval()


GEN_DIR = f'generated/{CHECKPOINT_SUFFIX}/'
os.makedirs(GEN_DIR, exist_ok=True)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('usage: python src/fairseq/gen_batch.py <prime_midi_file> <prime_measure_count> <prime_chord_count> <gen_count>')
        exit(0)
    midi_name = sys.argv[1].split('/')[-1][:-4]
    max_measure_cnt = int(sys.argv[2])
    max_chord_measure_cnt = int(sys.argv[3])
    prime, ins_label = process_prime_midi(sys.argv[1], max_measure_cnt, max_chord_measure_cnt)
    gen_cnt = int(sys.argv[4])
    for i in range(gen_cnt):
        while(True):
            try:
                generated, ins_logits = gen_one(m, prime, MIN_LEN = 1024)
                break
            except Exception as e:
                print(e)
                continue
        trk_ins_map = get_trk_ins_map(generated, ins_logits)
        note_seq = get_note_seq(generated, trk_ins_map)
        #print(f'{len(note_seq)} notes generated.')
        #print(note_seq) 
        timestamp = time.strftime("%m-%d_%H-%M-%S", time.localtime()) 
        # note_seq_to_midi_file(note_seq, f'{GEN_DIR}{midi_name}_prime{max_measure_cnt}_chord{max_chord_measure_cnt}_{timestamp}.mid')
        note_seq_to_midi_file(note_seq, f'output.mid')