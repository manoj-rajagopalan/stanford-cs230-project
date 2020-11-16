#!/bin/bash

conda_env=$(conda info | grep 'active environment' | cut -d ':' -f 2 | xargs)
if [[ "x${conda_env}x" != "xpytorch_latest_p37x" ]]; then
	echo "Cannot run without conda env: conda activate pytorch_latest_p37"
	exit 1
fi

PYTHONPATH=$PWD:$PYTHONPATH \
python3 luo16/train_test.py --data-path kitti_2015 --exp-name plumbing --batch-size 10 --learning-rate 0.01 --reduction-factor 1 --phase both --patch-size 13 --test-all TEST_ALL --max-batches 10

