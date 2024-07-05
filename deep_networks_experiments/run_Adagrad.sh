#!/bin/bash

source deep_networks_experiments/settings.sh

python run_deep_network.py \
    --optimizer "Adagrad" \
    --total_trials ${TOTAL_TRIALS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE}\
    --learning_rates 1e-4 5e-4 5e-3 1e-3 5e-2 1e-2 5e-1 1e-1  \
    --output_dir ${SAVEDIR} \
    --train_data_loc "data/cifar10_csv/cifar10_train.csv"\
    --test_data_loc "data/cifar10_csv/cifar10_test.csv" \
	--decay_learning_rate 

python run_deep_network.py \
    --optimizer "Adagrad" \
    --total_trials ${TOTAL_TRIALS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE}\
    --learning_rates 1e-4 5e-4 5e-3 1e-3 5e-2 1e-2 5e-1 1e-1  \
    --output_dir ${SAVEDIR} \
    --train_data_loc "data/cifar10_csv/cifar10_train.csv"\
    --test_data_loc "data/cifar10_csv/cifar10_test.csv" 