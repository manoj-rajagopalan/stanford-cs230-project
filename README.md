# CS230 Fall 2020
Implementation of a U-Net architecture to refine disparities calculated apriori by a Siamese architecture.

Implementation of the Siamese network with inner-product cost volume construction modeled on the paper

  W.Luo, A.G.Schwing, R.Urtasun, "Efficient Deep Learning for Stereo Matching", CVPR 2016.

Original code was written in TensorFlow 1.x.
We completely rewrote it in PyTorch 1.7 (luo16/ subdir) and refactored our implementation to include our attempt to use an alternate cost volume construction (see module/feature_vec_diff_norm_sqr.py)

The disparityFilter/ subdir contains our U-Net based post-processing implementation.

The run/ subdir contains scripts to invoke the siamese network for training. Run them from this dir.

The disparityFilter/ subdir is fully self-contained ... invoke it separately.

# Contributors
- Gordon Alexander Charles
- Manoj Rajagopalan

