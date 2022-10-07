
setup_linux:
#	Requirements will fail if pytorch doesn't already exist :-/
	pip install torch==1.12.1
	pip install -r requirements.txt

setup_osx:
#	Some of the python libs need openmp to compile and
# 	llvm now includes openmp?
	brew install llvm
	brew install libomp
# fast_transformers requires ninja
	brew install ninja
#	Requirements will fail if pytorch doesn't already exist :-/
	pip install torch==1.12.1
	pip install -r requirements.txt

bpe_exe:
	cd src/musicBPE; \
	g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o ../../music_bpe_exec

####################################################

test_run:
	export CUDA_VISIBLE_DEVICES=""; \
		python3 src/fairseq/gen_batch.py test.mid 5 0 1

####################################################
train:
	ls -alFh data/midis
	python3 src/preprocess/preprocess_midi.py
#	python3 src/preprocess/get_bpe_data.py
	python3 src/fairseq/make_data.py
	sh train_linear_chord.sh

docker:
	docker build . -t symphonynet

docker-run:
	docker run --gpus all symphonynet
