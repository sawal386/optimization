#!/bin/bash

source feed_forward_experiments/settings.sh

bash feed_forward_experiments/run_SAGD.sh
bash feed_forward_experiments/run_SAGD1.sh
bash feed_forward_experiments/run_SAGD2.sh
bash feed_forward_experiments/run_SGD.sh
bash feed_forward_experiments/run_Adagrad.sh
bash feed_forward_experiments/run_Adam.sh
bash feed_forward_experiments/run_Adam1.sh
bash feed_forward_experiments/run_Adam2.sh
