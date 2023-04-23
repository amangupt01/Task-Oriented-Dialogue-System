import jsonlines
import random

# Set the input and output file paths
input_file = 'data/train.jsonl'
dev_file = 'data/dev.jsonl'
output_file_all = 'data/train_dev.jsonl'
output_file_train = 'data/train_dev_80.jsonl'
output_file_test = 'data/train_dev_20.jsonl'

# Set the train-test split ratio
train_ratio = 0.8

# Read in the input file and shuffle the data
with jsonlines.open(input_file) as reader:
    with jsonlines.open(dev_file) as reader2:
        data = [line for line in reader]
        data.extend([line for line in reader2])
        random.shuffle(data)

# Split the data into train and test sets
split_idx = int(len(data) * train_ratio)
train_data = data[:split_idx]
test_data = data[split_idx:]

# Write the train data to a new file
with jsonlines.open(output_file_all, mode='w') as writer:
    for line in data:
        writer.write(line)

with jsonlines.open(output_file_train, mode='w') as writer:
    for line in train_data:
        writer.write(line)

# Write the test data to a new file
with jsonlines.open(output_file_test, mode='w') as writer:
    for line in test_data:
        writer.write(line)
