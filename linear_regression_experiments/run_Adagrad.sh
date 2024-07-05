#!/bin/bash

source linear_regression_experiments/settings.sh

echo "Running Linear Regression, Adagrad"
python run_linear_regression.py \
    --n_data ${DATA_SIZE} \
    --n_dim ${DATA_DIM} \
    --optimizer "Adagrad" \
    --total_trials ${TOTAL_TRIALS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --learning_rates 0.5 1 2 4 8 16 32 64 128\
    --output_dir ${SAVEDIR} \
    --batch_size ${BATCH_SIZE} \
    --average_size_plot ${AVERAGE_SIZE}
