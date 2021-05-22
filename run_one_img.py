import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
# import matplotlib.pyplot as plt
import numpy as np
import torchvision
import argparse
from torchvision.io import read_image
import torchvision.transforms as transforms
import open3d as o3d
import os
from task1and2 import primary as open2d

import json


from unit import CustomImageDataset
from unit import Net

def primary(path_depth, usegpu, PATH):

    # plt.figure(figsize=(15,5))
    if usegpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'


    pcd = o3d.io.read_point_cloud(path_depth)
    out_arr = np.asarray(pcd.points)

    Cin, D, H, W, A = 1, 20, 224, 224, 224

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

    out = torch.zeros((1, Cin ,D ,H ,W))
    for i in range(len(out_arr)):
        N0 = int((A-1) * out_arr[i,0])
        N1 = int((A-1) * out_arr[i,1])
        N2 = int((D-1) * out_arr[i,2])
        out[0, 0, N2, N0, N1] += 1

    out = out/torch.max(out)


    net = Net().to(device)
    net.load_state_dict(torch.load(PATH))

    outputs1, outputs2, outputs3 = net(out)

    _, predicted1 = torch.max(outputs1, 1)
    _, predicted2 = torch.max(outputs2, 1)
    _, predicted3 = torch.max(outputs3, 1)

    print('Predicted 1: ', ' '.join('%5s' % predicted1[j].cpu().item()
                                  for j in range(1)))
    print('Predicted 2: ', ' '.join('%5s' % predicted2[j].cpu().item()
                                  for j in range(1)))
    print('Predicted 3: ', ' '.join('%5s' % predicted3[j].cpu().item()
                                  for j in range(1)))
    doorsdict = {0:'OPEN', 1:'SEMI', 2:'CLOSED', 3:'UNKNOWN'}
    doorsdict2 = {0:'UNKNOWN', 1:'human', 2:'wear', 3:'limb', 4:'other'}

    bbox, shirina = open2d(path_depth)
    print(bbox)
    print(shirina)

    if len(bbox)>0:
        data = {'figures':[
            {"object":doorsdict2[predicted2.cpu().item()] ,
             "geometry":{'position':{'x':bbox[0][0],'y':bbox[0][1],'z':bbox[0][2] },
                         'rotation':{'x':0, 'y':0, 'z':0},
                         'dimensions':{'x':shirina[0][0],'y':shirina[0][1],'z':shirina[0][2] }},
             "door":doorsdict[predicted1.cpu().item()]
             }
        ]
        }
    else:
        data = {'figures':[
            {"object":doorsdict2[predicted2.cpu().item()] ,
             "geometry":{'position':{}, 'rotation':{}, 'dimensions':''},
             "door":doorsdict[predicted1.cpu().item()]
             }
        ]
        }

    with open('/home/neptun/Документы/Хакатон/network/train/js/'+path_depth.split(sep='/')[-1]+'.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=str,  default="/home/neptun/Документы/Хакатон/data/point_cloud_train/clouds_tof/cloud_0_0010.pcd", help="path to input depth image")
    parser.add_argument("--gpu", type=bool,  default=False, help="use gpu")
    parser.add_argument("--weight", type=str,  default='./rjd_3d_best_3.pth', help="PATH to weight net")
    args = parser.parse_args()
    primary(args.depth, args.gpu, args.weight)