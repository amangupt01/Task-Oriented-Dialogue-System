train:
	bash run_model.sh train data/train.jsonl data/dev.jsonl

test:
	bash run_model.sh test data/test.jsonl outputfile.txt
	