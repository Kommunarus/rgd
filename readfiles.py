import numpy as np
import open3d as o3d
import os
import math

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})


path1 = './data/point_cloud_train/clouds_tof'
# path1 = './data/point_cloud_train/clouds_stereo'
listfile = os.listdir(path1)

for ii in range(len(listfile)):
    p = os.path.join(path1, listfile[ii])
    pcd = o3d.io.read_point_cloud(p)
    # o3d.visualization.draw_geometries([pcd])
    out_arr = np.asarray(pcd.points)

    minx, maxx = np.min(out_arr[:, 0]), np.max(out_arr[:, 0])
    miny, maxy = np.min(out_arr[:, 1]), np.max(out_arr[:, 1])
    minz, maxz = np.min(out_arr[:, 2]), np.max(out_arr[:, 2])
    # print(minx, miny, minz)
    # print(maxx, maxy, maxz)

    out_arr[:, 0] = (out_arr[:, 0] - minx) / (maxx - minx)
    out_arr[:, 1] = (out_arr[:, 1] - miny) / (maxy - miny)
    out_arr[:, 2] = (out_arr[:, 2] - minz) / (maxz - minz)


    img_height = 480
    img_width = 640
    is_data = False
    min_d = 0
    max_d = 0
    img_depth = np.zeros((img_height, img_width), dtype='f8')
    for i in range(len(out_arr)):
        line = out_arr[i]
        d = float(line[2])
        row = int((img_height-1)*line[0])
        col = int((img_width-1)*line[1])
        img_depth[row, col] = d
        # min_d = min(d, min_d)
        # max_d = max(d, max_d)

    # max_min_diff = max_d - min_d


    # def normalize(x):
    #     return 255 * (x - min_d) / max_min_diff

    # normalize = np.vectorize(normalize, otypes=[np.float])
    # img_depth = normalize(img_depth)

    _fig, ax = plt.subplots()
    plt.imshow(img_depth, cmap='gray_r')
    _fig.savefig(p.replace('point_cloud_train','map').replace('pcd','jpg'))
    plt.clf()
    plt.cla()
    # plt.show()