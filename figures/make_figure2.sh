#!/bin/bash

source figures/settings.sh

python trajectory_analyzer.py \
    --tolerance 1e-7 \
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure2" \
    --learning_rate 1 \
    --n_iterations 100\
    --use_log
