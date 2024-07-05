#!/bin/bash

source linear_regression_experiments/settings.sh

echo "Running Linear Regression, SAGD"
python run_linear_regression.py \
    --n_data ${DATA_SIZE} \
    --n_dim ${DATA_DIM} \
    --optimizer "SAGD" \
    --total_trials ${TOTAL_TRIALS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --learning_rates 0.005 0.01 0.05 0.1 0.5 1 2\
    --output_dir ${SAVEDIR} \
    --batch_size ${BATCH_SIZE} \
    --average_size_plot ${AVERAGE_SIZE}
