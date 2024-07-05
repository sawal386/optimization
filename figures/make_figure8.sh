#!/bin/bash
source figures/settings.sh

pkl_files_dir="${MODEL_OUTPUT_DIR}/deep_networks_experiments_alexnet"
echo ${pkl_files_dir}
echo ${MODEL_OUTPUT_DIR}

echo "making combined plot"
python compare_multiple_performances.py \
    --pkl_locations "${pkl_files_dir}/train_costs_epoch" "${pkl_files_dir}/test_costs" "${pkl_files_dir}/train_accuracies_epoch" "${pkl_files_dir}/test_accuracies" \
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure8" \
    --x_label "Epochs" \
    --y_label_types "training cost" "testing cost" "training error" "testing error" \
    --titles "" "" "" ""\
    --y_limit 1e-2 5e0 9e-1 5e0 1e-3 3e0 2e-1 9e-1 \
    --plot_shades \
    --take_log_error
