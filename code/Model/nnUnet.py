import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is NCHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class nnUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(nnUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        n_filters = [32, 64, 128, 256, 512]

        self.inc = DoubleConv(n_channels, n_filters[0])
        self.down1 = Down(n_filters[0], n_filters[1])
        self.down2 = Down(n_filters[1], n_filters[2])
        self.down3 = Down(n_filters[2], n_filters[3])
        self.down4 = Down(n_filters[3], n_filters[4])
        self.up1 = Up(n_filters[4], n_filters[3])
        self.up2 = Up(n_filters[3], n_filters[2])
        self.up3 = Up(n_filters[2], n_filters[1])
        self.up4 = Up(n_filters[1], n_filters[0])
        self.outc = OutConv(n_filters[0], n_classes)
        self.relu=nn.ReLU()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        print(x.size())
        out = self.relu(self.outc(x))
        
        print(out.size())
        cell_sum=torch.sum(torch.sum(out,dim=3),dim=2)
        return out,cell_sum
def creat():
    model=nnUNet()
    loss=nn.MSELoss()
    if torch.cuda.is_available():
        model=model.cuda()
        loss=loss.cuda()
    optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    return model,loss,optimizer,scheduler