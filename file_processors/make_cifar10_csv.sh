#!/bin/bash

echo "Generating CIFAR 10 training csv file"

python cifar10_processor.py \
    --base_loc "/Users/sawal/Downloads/cifar-10-batches-py" \
    --file_names "data_batch_1" "data_batch_2" "data_batch_3" "data_batch_4" "data_batch_5"\
    --output_loc "/Users/sawal/Desktop/optimization/data/cifar10_csv"\
    --output_name "cifar10_train"


echo "Generating CIFAR 10 testing csv file"

python cifar10_processor.py \
    --base_loc "/Users/sawal/Downloads/cifar-10-batches-py" \
    --file_names "test_batch"\
    --output_loc "/Users/sawal/Desktop/optimization/data/cifar10_csv"\
    --output_name "cifar10_test"
