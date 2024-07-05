#!/bin/bash

source linear_regression_experiments/settings.sh

echo "Running Linear Regression, SGD"

python run_linear_regression.py \
    --n_data ${DATA_SIZE} \
    --n_dim ${DATA_DIM} \
    --optimizer "SGD" \
    --total_trials ${TOTAL_TRIALS} \
    --base_seed ${SEED} \
    --epochs ${EPOCHS} \
    --learning_rates 1e-5 5e-5 0.0001 0.0005 0.001 0.005 \
    --output_dir ${SAVEDIR} \
    --batch_size ${BATCH_SIZE} \
    --average_size_plot ${AVERAGE_SIZE} 
    #--decay_learning_rate
