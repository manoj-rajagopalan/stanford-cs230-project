#!/bin/bash

conda_env=$(conda info | grep 'active environment' | cut -d ':' -f 2 | xargs)
if [[ "x${conda_env}x" != "xpytorch_latest_p37x" ]]; then
	echo "Cannot run without conda env: conda activate pytorch_latest_p37"
	exit 1
fi

PYTHONPATH=$PWD:$PYTHONPATH \
python3 luo16/train_test.py \
	--data-path kitti_2015 \
	--results-dir /cs230-datasets/proj/results \
	--exp-name inner_product_baseline \
	--cost-volume-method inner_product_softmax \
	--batch-size 128 \
	--learning-rate 0.01 \
	--reduction-factor 1 \
	--phase both \
	--patch-size 37 \
	--num-iterations 1 \
	--test-all TEST_ALL

