from module.feature_vec_inner_product import FeatureVecInnerProduct

import torch.nn

class InnerProductNetwork(torch.nn.Module):
    """
        Implementation of the paper "Efficient Stereo ..."
    """

    def __init__(self, siamese_network):
        super(InnerProductNetwork, self).__init__()
        self.siamese_network = siamese_network
        self.feature_vec_inner_product = FeatureVecInnerProduct()

    def forward(self, left_patch, right_patch):
        left_feature, right_feature = self.siamese_network(left_patch, right_patch)
        feature_vec_inner_product = self.feature_vec_inner_product(left_feature, right_feature)
        return feature_vec_inner_product

# /class InnerProductNetwork