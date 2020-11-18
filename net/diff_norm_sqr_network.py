from siamese_network import SiameseNetwork
from module.feature_vec_diff_norm_sqr import FeatureVecDiffNormSqr

import torch.nn

class DiffNormSqrNetwork(torch.nn.Module):
    """
        Implementation of the paper "Efficient Stereo ..."
    """

    def __init__(self, siamese_network):
        super(DiffNormSqrNetwork, self).__init__()
        self.siamese_network = siamese_network
        self.feature_vec_inner_product = FeatureVecDiffNormSqr(self.siamese_network.FEATURE_VECTOR_SIZE)

    def forward(self, left_patch, right_patch):
        left_feature, right_feature = self.siamese_network(left_patch, right_patch)
        feature_vec_diff_norm_sqr = self.feature_vec_diff_norm_sqr(left_feature, right_feature)
        return feature_vec_diff_norm_sqr

# /class DiffNormSqrNetwork