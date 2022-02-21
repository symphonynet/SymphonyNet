# SymphonyNet
## training pipeline
- put your midi files into data/midis/
- run src/preprocess/preprocess_midi.py
- (optional) if you want to train the model with music bpe, then run src/preprocess/get_bpe_data.py
- run src/fairseq/make_data.py
- run src/fairseq/{model_script} to train your own model!
## generate pipeline
- put your checkpoint into ckpt/ or [download our pretrained model](https://drive.google.com/file/d/1xpkj_qN4MdLRkBdCXmfGjuWWjnTN1Og0/view?usp=sharing)
- run gen_batch.py with your prime midi file
