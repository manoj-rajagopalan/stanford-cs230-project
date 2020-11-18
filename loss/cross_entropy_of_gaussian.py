import torch
import torch.nn

class CrossEntropyOfGaussian(torch.nn.Module):
    """
      Interprets the Gaussian of the input as probability of the classes.
      That is, for each inner-product-value r^2, it interprets exp(-r^2) as a probability.
      Note that r^2 already includes a metric-tensor (inverse of gaussian's covariance).
      Computes the multi-class cross-entropy loss based on this and the given labels.
    """

    def __init__(self):
        super(CrossEntropyOfGaussian, self).__init__()

    def forward(self, feature_vec_diff_norm_sqr, labels):
        """Forward-propagates the network.

        Legend:
          B : number of mini-batches
          D : disparity range (eg: 201)

        Args:
          feature_vec_diff_norm_sqr (Tensor): output from FeatureVecDiffNormSqr module
                                              (B,D) dimensional
          labels (Tensor): the "ground truth" of the expected disparity for the
                           patches of dimensions (B, D)

        Returns:
          loss : scalar loss value

        Notes:
          * Each sample is a pixel
          * At input, left-patch is a receptive field around the sample pixels
            Right-patch is a larger receptive field (accounting for disparity)
          * This function is at the end of the network.
            By this point, each pixel is mapped to a feature vector for the left-patch
            and D feature-vectors (one per disparity candidate) for the right-patch.
          * There are B mini-batches of such data.           
        """

        # pred = torch.exp(-feature_vec_diff_norm_sqr / variance)
        # log_pred = torch.log(pred)
        log_pred = -feature_vec_diff_norm_sqr
        loss = -1.0 * torch.sum(labels * log_pred) / log_pred.size(0)

        return loss

# /class CrossEntropyOfGaussian
