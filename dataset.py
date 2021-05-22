import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
import os
import pandas as pd
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2 as cv
import pickle as pkl
import open3d as o3d
import numpy as np


rootp = '../../data/point_cloud_train/clouds_stereo'
listfile = os.listdir(rootp)
Cin, D, H, W, A = 1, 20, 224, 224, 224

rootf = f'/home/neptun/Документы/Хакатон/network/dataset/'
with open(os.path.join(rootf, 'label.txt'), 'r') as f:
    la = f.readlines()

for fi in la:
# for file in listfile:
    file = fi.split(sep='\t')[0]
    p = os.path.join(rootp, file)
    pcd = o3d.io.read_point_cloud(p)
    out_arr = np.asarray(pcd.points)
    if len(out_arr) == 0:
        continue


    minx, maxx = np.min(out_arr[:,0]), np.max(out_arr[:,0])
    miny, maxy = np.min(out_arr[:,1]), np.max(out_arr[:,1])
    minz, maxz = np.min(out_arr[:,2]), np.max(out_arr[:,2])
    # print(minx, miny, minz)
    # print(maxx, maxy, maxz)

    out_arr[:,0] = (out_arr[:,0] - minx)/(maxx - minx)
    out_arr[:,1] = (out_arr[:,1] - miny)/(maxy - miny)
    out_arr[:,2] = (out_arr[:,2] - minz)/(maxz - minz)
    # minx, maxx = np.min(out_arr[:,0]), np.max(out_arr[:,0])
    # miny, maxy = np.min(out_arr[:,1]), np.max(out_arr[:,1])
    # minz, maxz = np.min(out_arr[:,2]), np.max(out_arr[:,2])
    # print(minx, miny, minz)
    # print(maxx, maxy, maxz)

    out = torch.zeros((Cin ,D ,H ,W))
    for i in range(len(out_arr)):
        N0 = int((A-1) * out_arr[i,0])
        N1 = int((A-1) * out_arr[i,1])
        N2 = int((D-1) * out_arr[i,2])
        out[0, N2, N0, N1] += 1

    out = out/torch.max(out)

    with open(os.path.join('/home/neptun/Документы/Хакатон/network/dataset', file.replace('pcd','pkl')), 'wb') as f:
        pkl.dump(out, f)
        print(file)
