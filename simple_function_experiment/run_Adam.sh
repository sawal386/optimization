#!/bin/bash

source simple_function_experiment/settings.sh

echo "Running Adam 0.99 on random data"
python optimize_simple_function.py \
    --n_data ${N_DATA_RANDOM} \
    --optimizer "Adam"\
    --generation_method "random"\
    --base_seed ${SEED} \
    --learning_rates 0.0001 0.0005 0.005 0.001 0.05 0.01 0.5 0.1 1 \
    --total_trials ${TOTAL_TRIALS}  \
    --output_dir ${SAVEDIR} \
    --beta 0.99

echo "Running Adam 0.999 on random data"
python optimize_simple_function.py \
    --n_data ${N_DATA_RANDOM} \
    --optimizer "Adam"\
    --generation_method "random"\
    --base_seed ${SEED} \
    --learning_rates 0.0001 0.0005 0.005 0.001 0.05 0.01 0.1 0.5 1 \
    --total_trials ${TOTAL_TRIALS}  \
    --output_dir ${SAVEDIR} \
    --beta 0.999

echo "Running Adam 0.9999 on random data"
python optimize_simple_function.py \
    --n_data ${N_DATA_RANDOM} \
    --optimizer "Adam"\
    --generation_method "random"\
    --base_seed ${SEED} \
    --learning_rates 0.0001 0.0005 0.005 0.001 0.05 0.01 0.5 0.1 1 \
    --total_trials ${TOTAL_TRIALS}  \
    --output_dir ${SAVEDIR} \
    --beta 0.9999

echo "Running Adam 0.99 on step data"
python optimize_simple_function.py \
    --n_data ${N_DATA_STEP} \
    --optimizer "Adam"\
    --generation_method "step"\
    --base_seed ${SEED} \
    --learning_rates 0.0001 0.0005 0.005 0.001 0.05 0.01 0.5 0.1 1 \
    --total_trials 1  \
    --output_dir ${SAVEDIR} \
    --beta 0.99

echo "Running Adam 0.999 on step data"
python optimize_simple_function.py \
    --n_data ${N_DATA_STEP} \
    --optimizer "Adam"\
    --generation_method "step"\
    --base_seed ${SEED} \
    --learning_rates 0.0001 0.0005 0.005 0.001 0.05 0.01 0.1 0.5 1\
    --total_trials 1  \
    --output_dir ${SAVEDIR} \
    --beta 0.999

echo "Running Adam 0.9999 on step data"
python optimize_simple_function.py \
    --n_data ${N_DATA_STEP} \
    --optimizer "Adam"\
    --generation_method "step"\
    --base_seed ${SEED} \
    --learning_rates 0.0001 0.0005 0.005 0.001 0.05 0.01 0.5 0.1 1 \
    --total_trials 1  \
    --output_dir ${SAVEDIR} \
    --beta 0.9999

