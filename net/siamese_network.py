import torch.nn as nn

class SiameseNetwork(nn.Module):
    """
        Siamese sub-network that extract features from left and right patches.
        Used inside InnerProductNetwork and DiffNormSqrNetwork.
    """

    FEATURE_VECTOR_SIZE = 64

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers for 37x37 input patches
        self.cnn1 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5),  # 1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=5),  # 2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=5),  # 3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5),  # 4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5),  # 5
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5),  # 6
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5),  # 7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5),  # 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5),  # 9 - no ReLu on Output
            nn.BatchNorm2d(64),

        )

    def forward(self, left_patch, right_patch):
        left_feature = self.cnn1(left_patch)
        right_feature = self.cnn1(right_patch)
        return left_feature, right_feature

