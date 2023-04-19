#!/bin/bash

# Parse command line arguments
if [ "$1" == "train" ]; then
    train_file=$2
    dev_file=$3
    python src/main.py --mode train --train_path $train_file --dev_path $dev_file

elif [ "$1" == "test" ]; then
    test_file=$2
    output_file=$3
    python src/main.py --mode test --test_path $test_file --output_path $output_file
else
    echo "Invalid argument: $1"
    echo "Usage: ./run_model.sh [train|test] [input_file] [output_file]"
    exit 1
fi
