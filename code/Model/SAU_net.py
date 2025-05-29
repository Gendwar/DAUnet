import torch
import torch.nn as nn
# epoch=350 400
class self_attention(nn.Module):
    def __init__(self,inch,outch):
        super().__init__()
        self.outch=outch
        self.inch =inch
        self.get_theta=nn.Conv2d(inch,outch,kernel_size=1)
        self.get_phi  =nn.Conv2d(inch,outch,kernel_size=1)
        self.get_g    =nn.Conv2d(inch,outch,kernel_size=1)
        self.softmax  =nn.Softmax(dim=-1)
        self.dropout  =nn.Dropout2d(0.5)
        self.outl     =nn.Sequential(nn.Conv2d(outch,inch,kernel_size=1),nn.ReLU(),nn.BatchNorm2d(inch))
    def forward(self,x):
        theta=self.get_theta(x).view(x.size()[0],self.outch,-1)
        phi =self.get_phi(x).view(x.size()[0],self.outch,-1)
        g   =self.get_g(x).view(x.size()[0],self.outch,-1)
        theta = theta.permute(0,2,1)
        theta_phi = self.softmax(torch.matmul(theta,phi))
        # print(theta_phi.size())
        theta_phi = self.dropout(theta_phi)
        g =g.permute(0,2,1)
        y  = torch.matmul(theta_phi,g)
        y = y.permute(0,2,1).contiguous()
        y=y.view(x.size()[0],self.inch,x.size()[2],x.size()[3])
        # print(y.size())
        out =x+self.outl(y)
        return out
class conv_block(nn.Module):
    """
    Double Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch))
    def forward(self, x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    def __init__(self,inch,outch):
        super().__init__()
        self.up=nn.Sequential(
            nn.ConvTranspose2d(inch,outch,kernel_size=2,stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outch)
        )
        self.conv=conv_block(2*outch,outch)
    def forward(self,upf,skip):
        upf=self.up(upf)
        out=self.conv(torch.cat((upf,skip),dim=1))
        return out
class SAUnet(nn.Module):
    def __init__(self,inch=3,outch=1):
        super().__init__()
        chlist=[32,64,128,256]
        self.pool =nn.MaxPool2d(2)
        self.en0=conv_block(inch,chlist[0])
        self.en1=conv_block(chlist[0],chlist[1])
        self.en2=conv_block(chlist[1],chlist[2])
        self.en3=conv_block(chlist[2],chlist[3])
        self.selfa=self_attention(chlist[3],chlist[3])
        self.de3=up_conv(chlist[3],chlist[2])
        self.de2=up_conv(chlist[2],chlist[1])
        self.de1=up_conv(chlist[1],chlist[0])
        self.out=nn.Sequential(nn.Conv2d(chlist[0],1,kernel_size=1),nn.ReLU(inplace=True))
    def forward(self,x):
        en0=self.en0(x)
        en1=self.en1(self.pool(en0))
        en2=self.en2(self.pool(en1))
        en3=self.en3(self.pool(en2))
        mid=self.selfa(en3)
        de3=self.de3(mid,en2)
        de2=self.de2(de3,en1)
        de1=self.de1(de2,en0)
        out=self.out(de1)
        cell_sum=torch.sum(torch.sum(out,dim=3),dim=2)
        return out,cell_sum
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu', param=0.01))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
def creat():
    model=SAUnet()
    model.initialize_weights()
    loss=nn.MSELoss()
    if torch.cuda.is_available():
        model=model.cuda()
        loss=loss.cuda()
    
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=0.001)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    return model,loss,optimizer,scheduler