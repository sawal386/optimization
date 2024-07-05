#!/bin/bash

source figures/settings.sh

python compare_performances.py \
    --pkl_file_loc "${MODEL_OUTPUT_DIR}/linear_regression_experiments/train_costs" \
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure4" \
    --x_label "t, iteration number" \
    --y_label_type "func" \
    --y_lim 1e-10 1e2\
    --average_size 5 \
    --take_log_error
