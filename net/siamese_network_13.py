import torch.nn as nn

class SiameseNetwork13(nn.Module):
    def __init__(self):
        super(SiameseNetwork13, self).__init__()

        # Setting up the Sequential of CNN Layers for 13x13 input patches
        self.cnn1 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5),  # 1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=5),  # 2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=5),  # 9 - no ReLu on Output
            nn.BatchNorm2d(64),

        )

    def forward(self, left_patch, right_patch):
        left_feature = self.cnn1(left_patch)
        right_feature = self.cnn1(right_patch)
        return left_feature, right_feature
