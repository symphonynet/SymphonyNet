
# Music BPE
This submodule is forked from https://github.com/glample/fastBPE, and adapted for music BPE.
See more details at the original repository.

## Installation

Compile with:
```
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o music_bpe_exec
```

## Usage:

### List commands
```
./music_bpe_exec
usage: music_bpe_exec <command> <args>

The commands supported by fastBPE are:

learnbpe nCodes input1 [input2]      learn BPE codes from one or two text files

```


### Learn codes
```
./music_bpe_exec learnbpe 40000 train.de train.en > codes
```

### Learn codes in preprocess/get_bpe_data.py
```
# First copy the executable file 'music_bpe_exec' to project's root directory
# !cp music_bpe_exec ../../
output_dir = 'data/bpe_res/'
with open(output_dir+'codes.txt', 'w') as stdout:
    with open(output_dir+'merged_voc_list.txt', 'w') as stderr:
        subprocess.run(['./musicbpe', 'learnbpe', f'{MERGE_CNT}', output_dir+'ori_voc_cnt.txt'], stdout=stdout, stderr=stderr)
```

