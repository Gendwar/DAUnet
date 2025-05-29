
import os
import cv2
import glob
import torch
import tifffile
import numpy as np
import torch.nn as nn
from PIL import Image
import pandas as pd
import torchvision.transforms as Trans
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset,TensorDataset
from xml.etree import ElementTree as ET
def get_map(labelp,r,sigma):
    # labelp: Path of the annotation file 
    # r and sigma: Gaussian convolution diameter and sigma

    tree = ET.parse(labelp)
    root = tree.getroot()
    v1 = root.findall('object')
    dot=np.zeros((256,256))
    k=256/2040
    # k=1
    for ob in v1:
        xmin=int(ob.find('bndbox').find('xmin').text)*k
        xmax=int(ob.find('bndbox').find('xmax').text)*k
        ymin=int(ob.find('bndbox').find('ymin').text)*k
        ymax=int(ob.find('bndbox').find('ymax').text)*k
        x=int((xmin+xmax)/2)
        y=int((ymin+ymax)/2)
        dot[y,x]=1
        dot = dot.astype(np.float32)
        den = cv2.GaussianBlur(dot,(r,r),sigma,borderType=0)*10000
def zipdata(data_path):
    # data_path : '../Data/img' path of MFN dataset
    dataset={}
    imgl=glob.glob(data_path+'/*.png')
    train_l,test_l=train_test_split(imgl,test_size=0.25)
    train_l,val_l=train_test_split(train_l,test_size=0.333)
    train_d=readdata(train_l)
    test_d  =readdata(test_l)
    val_d = readdata(val_l)
    dataset['train']=train_d
    dataset['test']=test_d
    dataset['val']=val_d
    return dataset     
def readdata(iml):
    img_l=[]
    den_l=[]
    num_l=[]
    data={}
    k=0
    for imp in iml:
        img=Image.open(imp)
        img=Trans.ToTensor()(img)
        denp=imp.replace('img','den').replace('.png','.tif')
        den=tifffile.imread(imp.replace('img','den').replace('.png','.tif'))
        num_l.append(np.sum(den)/10000)
        den=torch.tensor(den).unsqueeze(0).to(torch.float32)
        img_l.append(img)
        den_l.append(den)
    data['img']=img_l
    data['den']=den_l
    data['num']=num_l
    return data