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
    
class Dense(nn.Module):
    def __init__(self,inch,midch=32,outch=8):
        super().__init__()
        self.l0=nn.Sequential(nn.Conv2d(inch,midch,kernel_size=1,bias=False),
                              nn.BatchNorm2d(midch),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(midch, outch, kernel_size=3,stride=1,padding=1,bias=False))
        self.l1=nn.Sequential(nn.Conv2d(inch+outch,midch,kernel_size=1,bias=False),
                            nn.BatchNorm2d(midch),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(midch, outch, kernel_size=3,stride=1,padding=1,bias=False))
        self.l2=nn.Sequential(nn.Conv2d(inch+2*outch,midch,kernel_size=1,bias=False),
                              nn.BatchNorm2d(midch),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(midch, outch, kernel_size=3,stride=1,padding=1,bias=False))
        self.l3=nn.Sequential(nn.Conv2d(inch+3*outch,midch,kernel_size=1,bias=False),
                              nn.BatchNorm2d(midch),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(midch, outch, kernel_size=3,stride=1,padding=1,bias=False))
    def forward(self,x):
        # print(x.size())
        x1=self.l0(x)
        x2=self.l1(torch.cat((x,x1),dim=1))
        x3=self.l2(torch.cat((x,x1,x2),dim=1))
        x4=self.l3(torch.cat((x,x1,x2,x3),dim=1))
        out=torch.cat((x,x1,x2,x3,x4),dim=1)
        # print(out.size())
        return out
class Up_Conv(nn.Module):
    def __init__(self,inch,outch):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv=nn.Sequential(nn.Conv2d(inch,outch,kernel_size=1,bias=False))
    def forward(self,x):
        out=self.up(x)
        out=self.conv(out)
        return out
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
class Multiscale_SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(Multiscale_SpatialAttentionModule, self).__init__()
        # self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        self.conv1=nn.Conv2d(in_channels=128,out_channels=1,kernel_size=1)
        self.conv2=nn.Conv2d(in_channels=128,out_channels=1,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=128,out_channels=1,kernel_size=5,padding=2)
        self.conv2d=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1)
    def forward(self, x):
        out1=self.conv1(x)
        out2=self.conv2(x)
        out3=self.conv3(x)
        out = torch.cat([out1, out2,out3], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
class CBAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = Multiscale_SpatialAttentionModule()
 
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
 
class DAUNet(nn.Module):
    def __init__(self,in_ch=3,out_ch=1):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_ch, 32,kernel_size=3,padding=1,stride=1,bias=False),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(inplace=True),
                                 Dense(32,32,8))
        self.down1=nn.Sequential(nn.MaxPool2d(2),
                                 Dense(64,32,16))
        self.down2=nn.Sequential(nn.MaxPool2d(2),
                                 Dense(128,32,32))
        self.up1  =up_conv(256,128) 
        self.upconv1=nn.Sequential(nn.Conv2d(256,64,kernel_size=1,bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   Dense(64,32,16),
                                   up_conv(128,64))
        self.out  =nn.Sequential(nn.Conv2d(128,64,kernel_size=1,bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                Dense(64,32,16),
                                CBAM(128),
                                nn.Conv2d(128,out_ch,kernel_size=1,bias=False),
                                nn.ReLU(inplace=True))
    def forward(self,x):
        e1=self.conv1(x)
        e2=self.down1(e1)
        e3=self.down2(e2)
        u1=self.up1(e3)
        u2=self.upconv1(torch.cat((e2,u1),dim=1))
        o=self.out(torch.cat((e1,u2),dim=1))
        cell_sum=torch.sum(torch.sum(o,dim=3),dim=2)
        return o,cell_sum
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
   
    model=DAUNet()
    model._initialize_weights()
    if torch.cuda.is_available():
        model=model.cuda()
        loss=loss.cuda()
    # optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=0.0001)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    return model,loss,optimizer,scheduler