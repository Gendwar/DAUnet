import os
import sys
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as Trans
from torch.utils.data import DataLoader,Dataset,TensorDataset
import Model.DAUNet as  Unet_DA

class CZI_dataset(Dataset):
    def __init__(self,data,trans_flag=True):
        super().__init__()
     
        self.img=data['img']
        self.den=data['den']
        self.cellnum=data['num']
        self.length=len(self.img)
        self.trans_flag=trans_flag
    def __getitem__(self, index):
        img=self.img[index]
        den=self.den[index]
        cellnum=self.cellnum[index]
        if self.trans_flag:
            img,den=self.trans(img,den)
        return img,den,cellnum
    def __len__(self):
        return self.length
    def trans(self,img,den):
        ph = random.randint(0, 1)
        pv = random.randint(0, 1)
        rp = random.choice([-45,-40,-35,-30,-25,-20-15,-10,-50,0,0,0,5,10,15,20,25,30,35,40,45])

        transf = Trans.Compose([Trans.RandomHorizontalFlip(ph),
                                Trans.RandomVerticalFlip(pv),
                                Trans.RandomRotation(degrees=(rp,rp))
                                ])
        return transf(img),transf(den)  
def get_dataload(datap):
    data=torch.load(datap)
    TrainD=CZI_dataset(data['train'])
    ValD  =CZI_dataset(data['val'],trans_flag=False)
    Trainload =DataLoader(TrainD,batch_size=12,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    Valload   =DataLoader(ValD,  batch_size=12,shuffle=False,num_workers=4,pin_memory=True,drop_last=False)
    
    return Trainload,Valload,[len(TrainD),len(ValD)]

def train(dp,model_flag='Unet_DA',epoch=400):
    Trainload,Valload,_=get_dataload(dp)
    net,loss,optimizer,scheduler=Unet_DA.create()
    valloss_min=float('inf')
    for ep in range(epoch):
        net.train()
        tloss=0
        for timg,tden,_ in Trainload:
            timg=timg.cuda()
            tden=tden.cuda()
            tden_pre,_=net(timg)
            ttloss=loss(tden_pre,tden)
            ttloss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tloss+=(float(ttloss))*int(timg.size()[0])
        scheduler.step()
        with torch.no_grad():
            net.eval()
            valloss=0
            for timg,tden,tnum in Valload:
                timg=timg.cuda()
                tden=tden.cuda()
                tnum=tnum.cuda()
                tden_pre,tnum_pre=net(timg)
                valloss+=(float(loss(tden_pre,tden))*int(timg.size()[0]))
            if valloss<valloss_min:
                best_model=copy.deepcopy(net.state_dict())
                valloss_min=valloss
    return best_model

