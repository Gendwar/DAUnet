import os
import cv2
import glob
import torch
import shutil
import tifffile
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import pandas as pd
import torchvision.transforms as Trans
from sklearn.model_selection import train_test_split
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader,Dataset,TensorDataset
import Model.DAUNet as  Unet_DA



def test_out(img,mdoel_parameters'):
    m_s=torch.load(mdoel_parameters,map_location='cpu')
    print(model_flag)
    net,_,_,_=Unet_DA.create()
    
    net.load_state_dict(m_s['best_model'])
    den_pre_l=[]
    with torch.no_grad():
        net.eval()    
        timg=img.cuda()
        tden_pre,cellnum_pre=net(timg)
        tden=tden_pre.cpu().squeeze().numpy()
        return teden
def get_match_PR(den,dot,thres,r):
#    den : Estimated distribution density maps
#    dot : annotation point
    p_a=0
    r_a=0
    dotm=0
    denm=0
    matchm=0
    point_l=[]
    den[den<thres]=0
    k=peak_local_max(den,min_distance=1,exclude_border=0)
    point_l=list(np.array(k))
    den_sum=len(point_l)
    h,w=dot.shape
    den_p=np.array(point_l)
    dot_p = np.where(dot>0)
    dot_p = np.array(dot_p).T
    dot_sum=np.sum(dot)
    Cmatrix=np.zeros((dot_sum,den_sum))
    max_r=np.sqrt(h**2+w**2)+1
    for i in range(dot_sum):
        for j in range(den_sum):
            Cmatrix[i,j]=np.sqrt((dot_p[i][0]-den_p[j][0])**2+(dot_p[i][1]-den_p[j][1])**2)
            Cmatrix[Cmatrix>r]=max_r
    match=scipy.optimize.linear_sum_assignment(Cmatrix)
    match=np.array(match).T

    match_r=[]
    out=np.zeros((dot.shape[0],dot.shape[1],3)).astype(np.uint8)
    den_m=[]
    dot_m=[]
    for i,j in match:
        if Cmatrix[i,j]<max_r:
            dot_m.append(i)
            den_m.append(j)
            match_r.append([i,j])

    Precision=float(len(match_r))/den_sum
    Recall   =float(len(match_r))/dot_sum

    return den_sum,len(match_r)