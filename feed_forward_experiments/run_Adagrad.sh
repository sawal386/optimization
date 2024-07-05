#!/bin/bash

source feed_forward_experiments/settings.sh

python run_feed_forward_net.py \
    --optimizer "Adagrad" \
    --total_trials ${TOTAL_TRIALS} \
    --n_hidden_units ${N_HIDDEN_UNITS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE}\
    --learning_rates 0.01 0.05 0.1 0.5 1 2\
    --output_dir ${SAVEDIR} \
    --train_data_loc "data/MNIST_CSV/mnist_train.csv"\
    --test_data_loc "data/MNIST_CSV/mnist_test.csv"\
	--decay_learning_rate 
    

python run_feed_forward_net.py \
    --optimizer "Adagrad" \
    --total_trials ${TOTAL_TRIALS} \
    --n_hidden_units ${N_HIDDEN_UNITS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE}\
    --learning_rates 0.01 0.05 0.1 0.5 1 2\
    --output_dir ${SAVEDIR} \
    --train_data_loc "data/MNIST_CSV/mnist_train.csv"\
    --test_data_loc "data/MNIST_CSV/mnist_test.csv"

