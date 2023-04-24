train:
	bash run_model.sh train data/train.jsonl data/dev.jsonl

tests:
	echo "hello"
	bash run_model.sh test data/sample_test.jsonl outputfile.txt
	