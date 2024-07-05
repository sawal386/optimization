#!/bin/bash

source figures/settings.sh

python compare_performances.py \
    --pkl_file_loc "${MODEL_OUTPUT_DIR}/simple_function_experiment_step/cost" \
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure1" \
    --x_label "t, iteration number" \
    --y_label_type "func" \
    --y_lim 0 1e2\
    --plot_shades
