import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class conv_block(nn.Module):
    """
    Double Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()
        n1 = 64
        chlist = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool = nn.MaxPool2d(2)
        self.Conv1 = conv_block(in_ch, chlist[0])
        self.Conv2 = conv_block(chlist[0], chlist[1])
        self.Conv3 = conv_block(chlist[1], chlist[2])
        self.Conv4 = conv_block(chlist[2], chlist[3])
        self.Conv5 = conv_block(chlist[3], chlist[4])
        self.Up5 = up_conv(chlist[4], chlist[3])
        self.Up_conv5 = conv_block(chlist[4], chlist[3])

        self.Up4 = up_conv(chlist[3], chlist[2])
        self.Up_conv4 = conv_block(chlist[3], chlist[2])

        self.Up3 = up_conv(chlist[2], chlist[1])
        self.Up_conv3 = conv_block(chlist[2], chlist[1])

        self.Up2 = up_conv(chlist[1], chlist[0])
        self.Up_conv2 = conv_block(chlist[1], chlist[0])

        self.out =nn.Sequential(nn.Conv2d(chlist[0], out_ch, kernel_size=1, stride=1, padding=0),
        nn.ReLU(inplace=True))
        
       # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        out = self.out(d2)
        cell_sum=torch.sum(torch.sum(out,dim=3),dim=2)
        #d1 = self.active(out)
        return out,cell_sum
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
def creat():
    model=U_Net()
    loss=nn.MSELoss()
    if torch.cuda.is_available():
        model=model.cuda()
        loss=loss.cuda()
    optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    return model,loss,optimizer,scheduler