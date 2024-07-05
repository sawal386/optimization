#!/bin/bash

source figures/settings.sh

python compare_performances.py  \
    --pkl_file_loc "${MODEL_OUTPUT_DIR}/logistic_regression_experiments/train_costs" \
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure5" \
    --x_label "t, iteration number" \
    --y_label_type "func" \
    --y_lim 1e-2 5e0 \
    --average_size 1 

: '
python compare_performances.py  \
    --pkl_file_loc "${MODEL_OUTPUT_DIR}/logistic_regression_experiments/train_accuracies" \
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure5 accuracy" \
    --x_label "# Iterations" \
    --y_label_type "Error" \
    --y_lim 0 1 \
    --average_size 1
    --plot_shades

python compare_performances.py  \
    --pkl_file_loc "${MODEL_OUTPUT_DIR}/logistic_regression_experiments_batch/train_costs" \
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure5_batch" \
    --x_label "# Iterations" \
    --y_label_type "func" \
    --y_lim 1e-2 1e1 \
    --average_size 1

python compare_performances.py  \
    --pkl_file_loc "${MODEL_OUTPUT_DIR}/logistic_regression_experiments_batch/train_accuracies" \
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure5_batch accuracy" \
    --x_label "# Iterations" \
    --y_label_type "Error" \
    --y_lim 0 1 \
    --average_size 1
'
