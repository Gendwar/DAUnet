import torch.nn as nn
import numpy as np
import torch
import random
import torch.nn.functional as F


class FCRN_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1),
                           nn.BatchNorm2d(32),   
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2))
        self.l2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1,stride=1),
                           nn.BatchNorm2d(64), 
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2))
        self.l3 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=1),
                           nn.BatchNorm2d(128), 
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2))
        self.l4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3,padding=1,stride=1),
                           nn.BatchNorm2d(512), 
                           nn.ReLU(inplace=True))
        self.l5 = nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                           nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3,padding=1,stride=1),
                           nn.BatchNorm2d(128),
                           nn.ReLU(inplace=True))
        self.l6 = nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                           nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,padding=1,stride=1),
                           nn.BatchNorm2d(64),
                           nn.ReLU(inplace=True))
        self.l7 = nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                           nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,padding=1,stride=1),
                           nn.BatchNorm2d(32),
                           nn.ReLU(inplace=True))
        self.l8 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3,stride=1,padding=1),
                                nn.ReLU(inplace=True) )
    def forward(self,x):
        x1 = self.l1(x)
        # print(x1.size())
        x2 = self.l2(x1)
        # print(x2.size())
        x3 = self.l3(x2)
        # print(x3.size())
        x4 = self.l4(x3)
        # print(x4.size())
        x5 = self.l5(x4)
        # print(x5.size())
        x6 = self.l6(x5)
        # print(x6.size())
        x7 = self.l7(x6)
        # print(x7.size())
        x8 = self.l8(x7)
        # print(x8.size())
        cell_sum=torch.sum(torch.sum(x8,dim=3),dim=2)
        return x8,cell_sum
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,nn.Linear)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
def creat():
    model=FCRN_A()
    model.initialize_weights()
    loss=nn.MSELoss()
    if torch.cuda.is_available():
        model=model.cuda()
        loss=loss.cuda()
    optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.99, weight_decay=0.0005)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    return model,loss,optimizer,scheduler