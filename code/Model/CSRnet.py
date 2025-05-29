import torch.nn as nn
import torch
from torchvision import models
# target = cv2.resize(target,(target.shape[1]/8,target.shape[0]/8),interpolation = cv2.INTER_CUBIC)*64
# epoch = 400
class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.up=nn.Sequential(nn.ConvTranspose2d(64,64,kernel_size=16,stride=8,padding=4),nn.ReLU(inplace=True),
                              nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                              nn.BatchNorm2d(64),
                              nn.ReLU(inplace=True))
        self.up=nn.Sequential(nn.ConvTranspose2d(64,64,kernel_size=16,stride=8,padding=4),
                              nn.BatchNorm2d(64),
                              nn.ReLU(inplace=True))
        self.output_layer = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1),nn.ReLU(inplace=True))
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        
        x = self.up(x)
        x = self.output_layer(x)
        return x,torch.sum(torch.sum(x,dim=3),dim=2)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                
def creat():
    model=CSRNet()
    # model.initialize_weights()
    loss=nn.MSELoss()
    if torch.cuda.is_available():
        model=model.cuda()
        loss=loss.cuda()
    optimizer=torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.95,weight_decay=5*1e-4)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    return model,loss,optimizer,scheduler