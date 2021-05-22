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
import torch.optim as optim
import random

D = 20
H, W = 224, 224
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomImageDataset(Dataset):
    def __init__(self, train, device):
        random.seed(21)
        self.root = f'/home/neptun/Документы/Хакатон/network/dataset/'
        self.device = device
        with open(os.path.join(self.root, 'label.txt'), 'r') as f:
            la = f.readlines()
        self.files_pkl = os.listdir(self.root)

        self.label = []
        ind = random.sample(range(len(la)), k=int(0.8*len(la)))
        for indx in range(len(la)):
            if indx in ind:
                if train:
                    self.label.append(la[indx])
            if not indx in ind:
                if not train:
                    self.label.append(la[indx])
        # print(train)
        # print(self.label)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        str = self.label[idx]
        da = str.split(sep='\t')
        fpkl, l1, l2, l3 = da


        with open(os.path.join(self.root, fpkl).replace('pcd','pkl'), 'rb') as f:
            image_pkl = pkl.load(f)


        return image_pkl.to(self.device), torch.tensor(int(l1)).to(self.device), torch.tensor(int(l2)).to(self.device),\
               torch.tensor(int(l3)).to(self.device)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # vgg_firstlayer=models.vgg16(pretrained = True).features[0] #load just the first conv layer
        # vgg=models.vgg16(pretrained = True).features[1:30] #load upto the classification layers except first conv layer
        #
        vgg = models.vgg16(pretrained=True)
        # w1=vgg_firstlayer.state_dict()['weight'][:,0,:,:]
        # w2=vgg_firstlayer.state_dict()['weight'][:,1,:,:]
        # w3=vgg_firstlayer.state_dict()['weight'][:,2,:,:]
        # w4=w1+w2+w3 # add the three weigths of the channels
        # w4=w4.unsqueeze(1)# make it 4 dimensional


        # first_conv.weigth=torch.nn.Parameter(w4, requires_grad=True) #initialize  the conv layer's weigths with w4
        # first_conv.bias=torch.nn.Parameter(vgg_firstlayer.state_dict()['bias'], requires_grad=True) #initialize  the conv layer's weigths with vgg's first conv bias


        # self.first_convlayer=first_conv #the first layer is 1 channel (Grayscale) conv  layer
        # self.vgg =nn.Sequential(vgg)

        # self.vgg =vgg
        self.features0 = nn.Conv3d(1, 3, (3,3,3), padding = (1,1,1)) #create a new conv layer
        self.features01 = nn.Conv3d(3, 64, (D,3,3), padding = (0,1,1)) #create a new conv layer
        torch.nn.init.xavier_uniform_(self.features0.weight)
        torch.nn.init.xavier_uniform_(self.features01.weight)
        self.features1 = vgg.features[1:]
        # self.avgpool = vgg.avgpool
        # self.classifier = vgg.classifier
        #
        # self.first_cl = vgg.classifier[:6]
        # self.last_fc = nn.Linear(4096, 2) #3
        # torch.nn.init.xavier_uniform_(self.last_fc.weight)
        # self.finish_cl = vgg.classifier[4:]

        # Дверь (0 - откр 1- в процессе закр , 2 -close? 3 - unknow)
        self.fc1 = nn.Linear(7*7*512, 1000)
        self.fc2 = nn.Linear(1000, 4)

        # Объект в портале (1 или 0)
        self.fc3 = nn.Linear(7*7*512, 1000)
        self.fc4 = nn.Linear(1000, 2)

        # Объект между платформой и вагоном (1 или 0)
        self.fc5 = nn.Linear(7*7*512, 1000)
        self.fc6 = nn.Linear(1000, 2)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        torch.nn.init.xavier_uniform_(self.fc6.weight)

    def forward(self, x):
        # x = self.first_convlayer(x)
        # x = self.vgg(x)
        # x = self.features(x)
        x = self.features0(x)
        x = self.features01(x)
        x = x.squeeze(dim = 2)
        x = self.features1(x)
        # x = self.avgpool(x)
        x = x.view(-1, 7 * 7 * 512)
        # x = self.classifier(x)
        # x = self.first_cl(x)
        # x = self.last_fc(x)

        # x = x.view(-1, 7 * 7 * 512)

        x1 = F.relu(self.fc1(x))
        x1 = self.fc2(x1)

        x2 = F.relu(self.fc3(x))
        x2 = self.fc4(x2)

        x3 = F.relu(self.fc5(x))
        x3 = self.fc6(x3)

        return x1, x2, x3

