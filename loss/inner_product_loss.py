import torch
from torch.nn import Module
import torch.nn.functional as F

class InnerProductLoss(Module):
    """
       To aid in training the inner product loss calculates the softmax with
       logits on a lables which have a probability distribution around the
       ground truth.  As a result the labels are not 0/1 integers, but floats.
    """

    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, left_feature, right_feature, labels):
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
          labels (Tensor): the "ground truth" of the expected disparity for the
                           patches of dimensions (1, disparity_range)

        Returns:
          left_patch (Tensor): sub-image left patch
          right_patch (Tensor): sub-image right patch
          labels (numpy array): the array used as "ground truth"

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
        # perform inner product of left and right features
        inner_product = torch.einsum('bf,bfd->bd', left_feature, right_feature)
        # peform the softmax with logits in two steps.  torch does not support
        # softmax with logits on float labels, so the calculation is broken
        # into calculating yhat and then the loss
        yhat = F.log_softmax(inner_product, dim=-1)
        loss = -1.0 * torch.einsum('bd,bd->b', yhat, labels).sum() / yhat.size(0)

        return loss

# /class InnerProductLoss()
