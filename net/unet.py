import torch.nn as nn

class UNet(nn.Module):

    def build_block(self, in_channels, out_channels, kernel_size, padding, upsample=False, relu=True):
        block = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      bias=True),
            # nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True),
            # nn.ReLU(inplace=True),
        ]
        if upsample:
            block.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        return nn.Sequential(*block)

    def __init__(self):
        super().__init__()

        self.down1 = self.build_block(in_channels=1, out_channels=64, kernel_size=3, padding=0)
        self.down2 = self.build_block(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.down3 = self.build_block(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.down4 = self.build_block(in_channels=128, out_channels=128, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.down5 = self.build_block(in_channels=128, out_channels=256, kernel_size=3, padding=0)
        self.down6 = self.build_block(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.down7 = self.build_block(in_channels=256, out_channels=512, kernel_size=3, padding=0)
        self.down8 = self.build_block(in_channels=512, out_channels=512, kernel_size=3, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.down9 = self.build_block(in_channels=512, out_channels=1024, kernel_size=3, padding=0)
        self.down10 = self.build_block(in_channels=1024, out_channels=512, kernel_size=3, padding=0, upsample=True)

        self.up1 = self.build_block(in_channels=1024, out_channels=512, kernel_size=3, padding=0)
        self.up2 = self.build_block(in_channels=512, out_channels=256, kernel_size=3, padding=0, upsample=True)

        self.up3 = self.build_block(in_channels=512, out_channels=256, kernel_size=3, padding=0)
        self.up4 = self.build_block(in_channels=256, out_channels=128, kernel_size=3, padding=0, upsample=True)

        self.up5 = self.build_block(in_channels=256, out_channels=128, kernel_size=3, padding=0)
        self.up6 = self.build_block(in_channels=128, out_channels=64, kernel_size=3, padding=0, upsample=True)

        self.up7 = self.build_block(in_channels=128, out_channels=64, kernel_size=3, padding=0)
        self.up8 = self.build_block(in_channels=64, out_channels=64, kernel_size=3, padding=0)

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # print("input size ", x.size())

        x = self.down1(x)
        # print("down 1 output size ", x.size())
        c1 = self.down2(x)
        # print("down 2 output size ", x.size())
        x = self.pool1(c1)
        # print("pool 1 output size ", x.size())

        # print("down 4 input size ", x.size())

        x = self.down3(x)
        c2 = self.down4(x)
        x = self.pool2(c2)

        # print("down 4 input size ", x.size())

        x = self.down5(x)
        c3 = self.down6(x)
        x = self.pool3(c3)

        # print("down 10 input size ", x.size())

        x = self.down7(x)
        c4 = self.down8(x)
        x = self.pool4(c4)

        # print("down 13 input size ", x.size())

        x = self.down9(x)
        # print("down 1 output size ", x.size())
        x = self.down10(x)
        # print("down 1 output size ", x.size())

        # print("C4 size ",c4.size())
        # print("down 15 output size ", x.size())
        # print("C4 reduced size ",c4[:,:,6:-6,6:-6].size())

        x = torch.cat([x, c4[:, :, 4:-4, 4:-4]], dim=1)
        x = self.up1(x)
        x = self.up2(x)

        x = torch.cat([x, c3[:, :, 16:-16, 16:-16]], dim=1)
        x = self.up3(x)
        x = self.up4(x)

        x = torch.cat([x, c2[:, :, 40:-40, 40:-40]], dim=1)
        x = self.up5(x)
        x = self.up6(x)

        x = torch.cat([x, c1[:, :, 88:-88, 88:-88]], dim=1)
        x = self.up7(x)
        x = self.up8(x)

        x = self.final_conv(x)
        return x

    def output_size(self, input_size):
        '''  Calculate output image size based upon input size and network configuration.
        #FIXME - constants should be moved to __init__ and should be parameters which
        configure the shape of the UNet for hyperparameter search.
        '''
        kernel_size = 3
        cnns_per_stage = 2
        depth = 4
        scale_factor_per_stage = 2

        f = (kernel_size - 1) * cnns_per_stage * depth
        y, x = input_size
        for _ in range(depth - 1):
            x = (x - (kernel_size - 1) * cnns_per_stage) / scale_factor_per_stage
            y = (y - (kernel_size - 1) * cnns_per_stage) / scale_factor_per_stage
            f = (f + (kernel_size - 1) * cnns_per_stage) * scale_factor_per_stage
        if x % 2 == 1:
            x_smaller = int(x)
            x_larger = x_smaller + 1
            for _ in range(depth - 1):
                x_smaller = (x_smaller + (kernel_size - 1) * cnns_per_stage) * scale_factor_per_stage
                x_larger = (x_larger + (kernel_size - 1) * cnns_per_stage) * scale_factor_per_stage
            print("Invalid Input width, valid sizes are ", x_smaller, "and", x_larger)
        if y % 2 == 1:
            y_smaller = int(y)
            y_larger = y_smaller + 1
            for _ in range(depth - 1):
                y_smaller = (y_smaller + (kernel_size - 1) * cnns_per_stage) * scale_factor_per_stage
                y_larger = (y_larger + (kernel_size - 1) * cnns_per_stage) * scale_factor_per_stage
            print("Invalid Input width, valid sizes are ", y_smaller, "and", y_larger)
        y, x = input_size

        return (y - f, x - f)

