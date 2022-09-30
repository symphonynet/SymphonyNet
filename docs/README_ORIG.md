# SymphonyNet
## Introduction
SymponyNet is an open-source project aiming to generate complex multi-track and multi-instrument music like symphony. 
Our method is fully compatible with other types of music like pop, piano, solo music..etc. 

<p>
<img src="./model_complete.jpg" alt="Schema." width="800px"></p>
Have fun with SymphonyNet !!
    
## Installation guide
We highly recommend users to run this project under `conda` environment.

#### Prepare the environment:
```
conda create -n your_env_name python=3.8
conda activate your_env_name

cd path_to_your_env
git clone this_project

cd SymphonyNet
cat requirements.txt | xargs -n 1 -L 1 pip install
``` 
The reason for using `cat requirements` is we find out the `pytorch-fast-transformers` package needs to be built upon torch, directly pip install requirements may cause `pytorch-fast-transformers` built error.

Note: Building `pytorch-fast-transformers` takes a while, please wait patiently.

## Training pipeline
### Step 1:
- Put your midi files into `data/midis/`

### Step 2:
- Run `python3 src/preprocess/preprocess_midi.py` under project root path

Quick note: The `preprocess_midi.py` multi-process all the Midis and convert them into a `raw_corpus.txt` file. In this
file, each line of encoded text represents a full song.

### Step 3 (optional):
- Run `python3 src/preprocess/get_bpe_data.py` if you want to train the model with Music BPE. More details about fast BPE
implementation could be found here [`Music BPE`](src/musicBPE/README.md).
- Set `BPE=1` in `config.sh` file

Note: We only provide `music_bpe_exec` file for linux system usage, if you are using MacOS or Windows, please re-compile 
the `music_bpe_exec` file [`here`](src/musicBPE/README.md) by following the instruction.

### Step 4:
- Run `python3 src/fairseq/make_data.py` to convert the `raw_corpus.txt` into binary file for fairseq and create `four
vocabularies` mentioned in the paper. 

### Step 5:
- Run `sh train_linear_chord.sh` to train your own model!

## Generation pipeline
- Put your checkpoint file or [download our pretrained model](https://drive.google.com/file/d/1xpkj_qN4MdLRkBdCXmfGjuWWjnTN1Og0/view?usp=sharing) into `ckpt/`
- Run `python3 src/fairseq/gen_batch.py test.mid 5 0 1` to generate one symphony MIDI conditioned on the first 5 measures of test.mid, with no constraints of chord progression.
- Or replace `test.mid` with your own prime MIDI and set how many measures of chords from the prime MIDI you may want to keep.
- We provide a [Google Colab file](https://colab.research.google.com/github/symphonynet/SymphonyNet/blob/main/play_symphonynet.ipynb) `play_symphonynet.ipynb`, where you could follow the generation guide. 

## License
SymphonyNet is released under the MIT license
