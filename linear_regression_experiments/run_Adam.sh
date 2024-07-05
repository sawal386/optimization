#!/bin/bash

source linear_regression_experiments/settings.sh

echo "Running Linear Regression, Adam 0.99"
python run_linear_regression.py \
    --n_data ${DATA_SIZE} \
    --n_dim ${DATA_DIM} \
    --optimizer "Adam" \
    --total_trials ${TOTAL_TRIALS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --learning_rates 0.001 0.005 0.01 0.05 0.1 0.5 1 2 4\
    --output_dir ${SAVEDIR} \
    --batch_size ${BATCH_SIZE} \
    --beta 0.99 \
    --average_size_plot ${AVERAGE_SIZE}

echo "Running Linear Regression, Adam 0.999"
python run_linear_regression.py \
    --n_data ${DATA_SIZE} \
    --n_dim ${DATA_DIM} \
    --optimizer "Adam" \
    --total_trials ${TOTAL_TRIALS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --learning_rates 0.001 0.005 0.01 0.05 0.1 0.5 1 2 4\
    --output_dir ${SAVEDIR} \
    --batch_size ${BATCH_SIZE} \
    --beta 0.999 \
    --average_size_plot ${AVERAGE_SIZE}

echo "Running Linear Regression, Adam 0.9999"
python run_linear_regression.py \
    --n_data ${DATA_SIZE} \
    --n_dim ${DATA_DIM} \
    --optimizer "Adam" \
    --total_trials ${TOTAL_TRIALS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --learning_rates 0.001 0.005 0.01 0.05 0.1 0.5 1 2 4\
    --output_dir ${SAVEDIR} \
    --batch_size ${BATCH_SIZE} \
    --beta 0.9999 \
    --average_size_plot ${AVERAGE_SIZE}

