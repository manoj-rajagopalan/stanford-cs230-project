# train_test.py Setup

Environment: for PyTorch 1.7 with Python3.7 (CUDA 11.0 and Intel MKL)
```
source activate pytorch_latest_p37
```
## NOTE:

The first time the command is run under anaconda it can take a while to start.

Invoke:

The following command line includes the more commonly used options.
```
python3 train_test.py --data-path kitti_2015 --exp-name test1 --batch-size 128 --learning-rate 0.01 --reduction-factor 1 --phase both --patch-size 37
```
**Note** `--patch-size` now supports 37x37 patches and 13x13 patches

**Note** `--test-all TEST_ALL` will result in the testing being run on all inputs.  This is intended to be used for generating
disparity data for use in training of a subsequent filtering network.

The following will show the complete list:
```
python3 train_test.py --help
```
Calling with no options will run the defaults, provided the kitti database is
in a directory named kitti_2015 in the same directory as `train_test.py`
```
python3 train_test.py --help
````
# train_test.py Output


**results/"experiment name":**
| name | description |
|------|--------------|
|images| directory of test images |
|model.pt | model_weights from training / for testing |
patch_locations.pkl  | patch location file - large, containing 11+M patch locations for 200 image.  Note that the files chosen for training / versus testing are random.  To run inference the same patch locations file used for testing is required.  This will automatically happen if the same experiment name is used for both training and testing. |
settings.log | capture of settings for the experiment |
stdout  | learning rate, training loss and validation loss are only printed to stdout and not captured in a log. Similarly, error per image and total testing error are only printed to stdout. |

# Pretrained

`results/pretrained` is a pretrained directory which can support running test
without having to run training.  The generated files, save the output from
testing have been modified to read only and a backup copy of the files is
in `results/pretrained/backup`.
