#!/bin/bash

source figures/settings.sh
: '
python compare_performances.py \
    --pkl_file_loc "${MODEL_OUTPUT_DIR}/feed_forward_experiments/train_costs_epoch" \
    --files_list "Adam_0.9_decay.pkl" "Adam_0.9.pkl" "Adagrad.pkl" "Adagrad_decay.pkl" "SAGD_0.01.pkl" "SGD.pkl" "SGD_decay.pkl"\
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure7_train_loss" \
    --x_label "Epochs" \
    --y_label_type "func" \
    --title "Training Loss" \
    --y_lim 1e-5 1e0

echo
python compare_performances.py \
    --pkl_file_loc "${MODEL_OUTPUT_DIR}/feed_forward_experiments/test_costs" \
    --files_list "Adam_0.9_decay.pkl" "Adam_0.9.pkl" "Adagrad.pkl" "Adagrad_decay.pkl" "SAGD_0.01.pkl" "SGD.pkl" "SGD_decay.pkl"\
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure7_test_loss" \
    --x_label "Epochs" \
    --y_label_type "func" \
    --title "Testing Loss" \
    --y_lim 1e-5 1e0

echo
python compare_performances.py \
    --pkl_file_loc "${MODEL_OUTPUT_DIR}/feed_forward_experiments/train_accuracies_epoch" \
    --files_list "Adam_0.9_decay.pkl" "Adam_0.9.pkl" "Adagrad.pkl" "Adagrad_decay.pkl" "SAGD_0.01.pkl" "SGD.pkl" "SGD_decay.pkl"\
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure7_train_accuracy" \
    --x_label "Epochs" \
    --y_label_type "Error" \
    --title "Training Error" \
    --y_lim 1e-5 1e-1 \
    --use_log


echo
python compare_performances.py \
    --pkl_file_loc "${MODEL_OUTPUT_DIR}/feed_forward_experiments/test_accuracies" \
    --files_list "Adam_0.9_decay.pkl" "Adam_0.9.pkl" "Adagrad.pkl" "Adagrad_decay.pkl" "SAGD_0.01.pkl" "SGD.pkl" "SGD_decay.pkl"\
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure7_test_accuracy" \
    --x_label "Epochs" \
    --y_label_type "Error" \
    --title "Testing Error" \
    --y_lim 1e-2 6e-2 \
    --use_log
    
'

echo "making combined plot"
python compare_multiple_performances.py \
    --pkl_locations "${MODEL_OUTPUT_DIR}/feed_forward_experiments_mnist/train_costs_epoch" "${MODEL_OUTPUT_DIR}/feed_forward_experiments_mnist/test_costs" "${MODEL_OUTPUT_DIR}/feed_forward_experiments_mnist/train_accuracies_epoch" "${MODEL_OUTPUT_DIR}/feed_forward_experiments_mnist/test_accuracies" \
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure7" \
    --x_label "Epochs" \
    --y_label_types "training cost" "testing cost" "training error" "testing error" \
    --titles "" "" "" ""\
    --y_limit 1e-5 1e0 5e-2 5e-1 1e-4 5e-1 1e-2 1e-1 \
    --plot_shades \
    --take_log_error

: '
echo "making combined plot"
python compare_multiple_performances.py \
    --pkl_locations "${MODEL_OUTPUT_DIR}/feed_forward_experiments_fmnist/train_costs_epoch" "${MODEL_OUTPUT_DIR}/feed_forward_experiments_fmnist/test_costs" "${MODEL_OUTPUT_DIR}/feed_forward_experiments_fmnist/train_accuracies_epoch" "${MODEL_OUTPUT_DIR}/feed_forward_experiments_fmnist/test_accuracies" \
    --output_dir "${FIGURE_OUTPUT_DIR}" \
    --output_name "figure7_combined_fmnist" \
    --x_label "Epochs" \
    --y_label_types "Cost" "Cost" "Error" "Error" \
    --titles "Training Cost" "Testing Cost" "Train Error" "Test Error"\
    --y_limit 1e-2 1e0 1e-1 5e-1 1e-2 5e-1 1e-1 5e-1 \
    --plot_shades
'



