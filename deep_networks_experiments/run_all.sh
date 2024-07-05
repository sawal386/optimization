#!/bin/bash

source deep_networks_experiments/settings.sh

bash deep_networks_experiments/run_Adam.sh
bash deep_networks_experiments/run_Adagrad.sh
bash deep_networks_experiments/run_SGD.sh
bash deep_networks_experiments/run_SAGD.sh

