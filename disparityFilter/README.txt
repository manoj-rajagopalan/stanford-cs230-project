disparity_filtering.py Setup
============================

Environment: for PyTorch 1.7 with Python3.7 (CUDA 11.0 and Intel MKL)

source activate pytorch_latest_p37

NOTE:
====

The first time the command is run under annaconda it can take a while to start.

Invoke:

The following command line includes the more commonly used options.

python disparity_filtering.py --batch-size 128 --learning-rate 0.01 --reduction-factor 1 --phase both --exp-name test_dir

The following will show the complete list:

python3 train_test.py --help

Calling with no options will run the defaults, provided the kitti database is
in a directory named kitti2015_plus in the same directory as disparity_filtering.py


train_test.py Output
====================

results/"experiment name":
images               directory of test images
model.pt             model_weights from training / for testing
patch_locations.pkl  patch location file - large, containing patch
                     locations for 200 images.
settings.log         capture of settings for the experiment

stdout               learning rate, training loss and validation loss are
                     only printed to stdout and not captured in a log.
                     Similarly, error per image and total testing error
                     are only printed to stdout.
