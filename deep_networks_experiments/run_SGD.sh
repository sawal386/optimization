#!/bin/bash

source deep_networks_experiments/settings.sh

python run_deep_network.py \
    --optimizer SGD \
    --total_trials ${TOTAL_TRIALS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE}\
    --learning_rates 0.01 0.05 0.125 0.25 0.5 1 2 4  \
    --output_dir ${SAVEDIR} \
    --train_data_loc "data/cifar10_csv/cifar10_train.csv"\
    --test_data_loc "data/cifar10_csv/cifar10_test.csv"
	
python run_deep_network.py \
    --optimizer SGD \
    --total_trials ${TOTAL_TRIALS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE}\
    --learning_rates 0.01 0.05 0.125 0.25 0.5 1 2 4 \
    --output_dir ${SAVEDIR} \
    --train_data_loc "data/cifar10_csv/cifar10_train.csv"\
    --test_data_loc "data/cifar10_csv/cifar10_test.csv"\
	--decay_learning_rate 

