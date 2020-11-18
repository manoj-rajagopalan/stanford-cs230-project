import torch
from torch.nn import Module
import torch.nn.functional as F

class CrossEntropyOfSoftmax(Module):
    """
       To aid in training the inner product loss calculates the softmax with
       logits on a lables which have a probability distribution around the
       ground truth.  As a result the labels are not 0/1 integers, but floats.
    """

    def __init__(self):
        super(CrossEntropyOfSoftmax, self).__init__()

    def forward(self, feature_vec_inner_prods, labels):
        """Calculate the loss describe above.

        Legend:
          B : number of mini-batches
          D : disparity range (eg: 201)

        Args:
          feature_vec_inner_prods (B x D Tensor): inner products of reference left feature-vector
                                                  with right "disparity-candidate" feature-vectors.
                                                  Output of FeatureVecInnerProduct module.
          labels (Tensor): the "ground truth" of the expected disparity for the
                           patches of dimensions (B, D)

        Returns:
          loss (scalar)
        """
        yhat = F.log_softmax(feature_vec_inner_prods, dim=-1)
        loss = -1.0 * torch.einsum('bd,bd->b', yhat, labels).sum() / yhat.size(0)

        return loss

# /class CrossEntropyOfSoftmax()
