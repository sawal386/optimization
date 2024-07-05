#!/bin/bash

source simple_function_experiment/settings.sh

echo "Running SGD on random data"
python optimize_simple_function.py \
    --n_data ${N_DATA_RANDOM} \
    --optimizer "SGD"\
    --generation_method "random"\
    --base_seed ${SEED} \
    --learning_rates  0.001 0.05 0.01 0.5 0.1 1 2 4 \
    --total_trials ${TOTAL_TRIALS} \
    --output_dir ${SAVEDIR} 

echo "Running SGD on step data"
python optimize_simple_function.py \
    --n_data ${N_DATA_STEP} \
    --optimizer "SGD"\
    --generation_method "step"\
    --base_seed ${SEED} \
    --learning_rates 0.001 0.05 0.01 0.5 0.1 1 2 4 \
    --total_trials 1 \
    --output_dir ${SAVEDIR}
