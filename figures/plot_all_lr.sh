#!/bin/bash

source figures/settings.sh

python plot_all_lr.py \
  -loc "${OUTPUT_DIR}/feed_forward_experiments/train_costs_128" \
  -o "${OUTPUT_DIR}/feed_forward_experiments/figures_128" \
  -x "# Data" \
  -y_type "func" \
  -lim 1e-5 1e1 \
  -log "True"\
  -ib "False"
