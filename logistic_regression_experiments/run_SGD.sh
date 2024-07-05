#!/bin/bash

source logistic_regression_experiments/settings.sh

echo "Running Logistic Regression, SGD"
python run_logistic_regression.py \
    --n_data ${DATA_SIZE} \
    --n_dim ${DATA_DIM} \
    --optimizer "SGD" \
    --bin_size ${BUCKET_SIZE}\
    --total_trials ${TOTAL_TRIALS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --learning_rates 0.01 0.05 0.1 0.5 1 2 4 8\
    --output_dir ${SAVEDIR} \
    --batch_size ${BATCH_SIZE}\
    --average_size_plot ${AVERAGE_SIZE}

