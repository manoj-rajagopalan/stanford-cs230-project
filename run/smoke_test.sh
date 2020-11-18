#!/bin/bash

# NOTE: run from parent dir

conda_env=$(conda info | grep 'active environment' | cut -d ':' -f 2 | xargs)
if [[ "x${conda_env}x" != "xpytorch_latest_p37x" ]]; then
	echo "Cannot run without conda env: conda activate pytorch_latest_p37"
	exit 1
fi

if [[ -z "$1" ]]; then
	echo "Usage: $(basename $0) <experiment name>"
	exit 1
fi

PYTHONPATH=$PWD:$PYTHONPATH \
python3 luo16/train_test.py \
	--data-path kitti_2015 \
	--result-dir /cs230-datasets/proj/checks \
	--exp-name $1 \
	--cost-volume-method inner_product_softmax \
	--batch-size 8 \
	--learning-rate 0.01 \
	--reduction-factor 1 \
	--phase both \
	--patch-size 13 \
	--max-batches 3 \
	--num-iterations 3 \
        --test-all TEST_ALL
