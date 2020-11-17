import torch
import torch.nn

class DiffNormGaussianLoss(torch.nn.Module):
    """
        First runs both patches through the siamese sub-network.
        Per-pixel it then does the following.
        It computes the difference between the left-patch feature vector (reference)
        and the right-patch feature vectors (one per disparity value).
        It uses the norm of these "diff" vectors in a gaussian to get a value between 0 and 1.
        It uses a 'variance' hyperparameter to tune the sharpness of this gaussian.
    """

    def __init__(self):
        super(DiffNormGaussianLoss, self).__init__()

    def forward(self, left_feature, right_feature, labels, variance):
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
        left_feature = torch.squeeze(left_feature)
        right_feature = torch.squeeze(right_feature)
        left_feature = left_feature.reshape_as(right_feature) # logically "copy" feature vector #Disparity-Bin times, per pixel
        diff_feature = right_feature - left_feature
        diff_norm_sqr = torch.einsum('bfd,bfd->bd', diff_feature, diff_feature)
        # pred = torch.exp(-diff_norm_sqr / variance)
        # log_pred = torch.log(pred)
        log_pred = -diff_norm_sqr / variance
        loss = -1.0 * torch.sum(labels * log_pred, dim=1)

        return loss

# /class SiameseNetworkGNPP
