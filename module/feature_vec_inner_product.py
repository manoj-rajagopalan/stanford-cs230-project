import torch
from torch.nn
import torch.nn.functional as F

class FeatureVecInnerProduct(torch.nn.Module):
    """
       To aid in training the inner product loss calculates the softmax with
       logits on a lables which have a probability distribution around the
       ground truth.  As a result the labels are not 0/1 integers, but floats.
    """

    def __init__(self):
        super(FeatureVecInnerProduct, self).__init__()

    def forward(self, left_feature, right_feature):
        """Calculate the loss describe above.

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
            inner_product (Tensor): (B,D) dimensional inner-product of left feature vector
                                    with right feature vectors

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
        inner_product = torch.einsum('bf,bfd->bd', left_feature, right_feature)
        return inner_product

# /class FeatureVecInnerProduct
