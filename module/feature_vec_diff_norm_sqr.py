import torch
import torch.nn

class FeatureVecDiffNormSqr(torch.nn.Module):
    """
        Computes Square-Norm of difference of feature vectors.
        Because the NN learns feature vectors, nothing is known about this "vector space" apriori.
        This module learns the Metric tensor for this inner-product space.
        Although the Metric is supposed to be symmetric-positive-definite for geometrically-correct
        inner-product spaces, we do not impose that constraint here: we let it be a square, dense matrix.
    """

    def __init__(self, feature_vec_size):
        super(FeatureVecDiffNormSqr, self).__init__()

        # Inverse "covariance matrix" for the Gaussian that will be computed in the loss function
        # Initialize to be symmetric. Hope that it stays this way.
        self.metric_tensor = torch.nn.Parameter(torch.Tensor(feature_vec_size, feature_vec_size))
        self.metric_tensor.data = 0.99 * torch.eye(feature_vec_size) + 0.01 * torch.ones([feature_vec_size, feature_vec_size])

    def forward(self, left_feature, right_feature):
        """Forward-propagates the network.

        Legend:
          B : number of mini-batches
          F : number of features (eg: 64)
          D : disparity range (eg: 201)

        Args:
          left_feature (Tensor): output of from the left patch passing through
                                 the Siamese network of dimensions (B, F, 1, 1)
          right_feature (Tensor): output of from the right patch passing through
                                  the Siamese network of dimensions
                                  (B, F, 1, D)

        Returns:
          diff_norm_sqr (Tensor): (B,D) dimensional output of value of v^T M v,
                                   per-disparity-candidate, where v is the vector difference
                                   between left and right features, and M is the learned metric
                                   tensor for this space.

        Notes:
          * Each sample is a pixel
          * At input, left-patch is a receptive field around the sample pixels
            Right-patch is a larger receptive field (accounting for disparity)
          * This function is at the end of the network.
            By this point, each pixel is mapped to a feature vector for the left-patch
            and D feature-vectors (one per disparity candidate) for the right-patch.
          * There are B mini-batches of such data.           
        """
        left_feature = left_feature.expand_as(right_feature) # logically "copy" feature vector #Disparity-Bin times, per pixel
        diff_feature = torch.squeeze(right_feature - left_feature)
        metric_diff = torch.einsum('if,bfd->bid', self.metric_tensor, diff_feature)
        diff_norm_sqr = torch.einsum('bfd,bfd->bd', diff_feature, metric_diff)

        return diff_norm_sqr

# /class FeatureVecDiffNormSqr
