import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
# import matplotlib.pyplot as plt
import numpy as np
import torchvision
import argparse
import json

from unit import CustomImageDataset
from unit import Net

def primary(dir, usegpu, PATH):

    # plt.figure(figsize=(15,5))
    if usegpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    # PATH = './rjd_net_2.pth'
    testing_data = CustomImageDataset(
        train=False, device=device
    )

    testloader = DataLoader(testing_data, batch_size=64, shuffle=False)

    net = Net().to(device)
    net.load_state_dict(torch.load(PATH))

    dataiter = iter(testloader)
    combi, l1, l2, l3 = dataiter.next()

    # print images
    # imshow(torchvision.utils.make_grid(rgb.type(torch.IntTensor)))

    outputs1, outputs2, outputs3 = net(combi)

    _, predicted1 = torch.max(outputs1, 1)
    _, predicted2 = torch.max(outputs2, 1)
    _, predicted3 = torch.max(outputs3, 1)

    print('GroundTruth 1: ', ' '.join('%5s' % l1[j].cpu().item() for j in range(len(testloader.dataset))))
    print('Predicted   1: ', ' '.join('%5s' % predicted1[j].cpu().item()
                                  for j in range(len(testloader.dataset))))
    print('\n')
    print('GroundTruth 2: ', ' '.join('%5s' % l2[j].cpu().item() for j in range(len(testloader.dataset))))
    print('Predicted   2: ', ' '.join('%5s' % predicted2[j].cpu().item()
                                  for j in range(len(testloader.dataset))))
    print('\n')
    print('GroundTruth 3: ', ' '.join('%5s' % l3[j].cpu().item() for j in range(len(testloader.dataset))))
    print('Predicted   3: ', ' '.join('%5s' % predicted3[j].cpu().item()
                                  for j in range(len(testloader.dataset))))

    doorsdict = {0:'OPEN', 1:'SEMI', 2:'CLOSED', 3:'UNKNOWN'}
    doorsdict2 = {1:'human', 0:'other'}

    for i, sttest in enumerate(testing_data.label):
        data = {'figures':[
            {"object":doorsdict2[predicted2[i].cpu().item()] ,
             "geometry":{},
             "door":doorsdict[predicted1[i].cpu().item()]
             }
        ]
        }
        namef = sttest.split(sep='\t')[0]
        with open('./js/'+namef+'.json', 'w') as f:
            json.dump(data, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,  default="4", help="path to input dir")
    parser.add_argument("--gpu", type=bool,  default=False, help="use gpu")
    parser.add_argument("--weight", type=str,  default='./rjd_3d_best_3.pth', help="PATH to weight net")
    args = parser.parse_args()
    primary(args.input, args.gpu, args.weight)