
test_run:
	export CUDA_VISIBLE_DEVICES=""; \
		python3 src/fairseq/gen_batch.py test.mid 5 0 1


docker:
	docker build . -t symphonynet

docker-run:
	docker run --gpus all symphonynet
