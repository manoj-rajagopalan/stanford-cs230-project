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

        Args:
          left_feature (Tensor): output of from the left patch passing through
                                 the Siamese network of dimensions (1, 1, 1, 64)
          right_feature (Tensor): output of from the right patch passing through
                                  the Siamese network of dimensions
                                  (1, 1, disparity_range, 64)
          lables (Tensor): the "ground truth" of the expected disparity for the
                           patches of dimensions (1, disparity_range)


        Returns:
          left_patch (Tensor): sub-image left patch
          right_patch (Tensor): sub-image right patch
          labels (numpy array): the array used as "ground truth"
        """
        left_feature = torch.squeeze(left_feature)
        # perform inner product of left and right features
        inner_product = torch.einsum('il,iljk->ik', left_feature, right_feature)
        # peform the softmax with logits in two steps.  torch does not support
        # softmax with logits on float labels, so the calculation is broken
        # into calculating yhat and then the loss
        yhat = F.log_softmax(inner_product, dim=-1)
        loss = -1.0 * torch.einsum('ij,ij->i', yhat, labels).sum() / yhat.size(0)

        return loss

# /class InnerProductLoss()
