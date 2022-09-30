
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

####################################################

test_run:
	export CUDA_VISIBLE_DEVICES=""; \
		python3 src/fairseq/gen_batch.py test.mid 5 0 1

####################################################

docker:
	docker build . -t symphonynet

docker-run:
	docker run --gpus all symphonynet
