#!/bin/bash

source figures/settings.sh

python compare_performances.py \
    --pkl_file_loc "${MODEL_OUTPUT_DIR}/simple_function_experiment_random/cost" \
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure3" \
    --x_label "t, iteration number" \
    --y_label_type "func" \
    --y_lim 0.2 1e1\
    --take_log_error \
    --measure "std"\
    --plot_shades
