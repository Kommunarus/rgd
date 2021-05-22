import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from matplotlib import colors
import matplotlib as mpl
import functools
import argparse
import open3d as o3d


S = 100 # ограничение контуров по площади для перона
S2 = 1000 # ограничение контуров по площади для щели

def convex_hull_graham(points):
    '''
    Returns points on convex hull in CCW order according to Graham's scan algorithm.
    By Tom Switzer <thomas.switzer@gmail.com>.
    '''
    TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

    def cmp(a, b):
        return (a > b) - (a < b)

    def turn(p, q, r):
        return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

    def _keep_left(hull, r):
        while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
            hull.pop()
        if not len(hull) or hull[-1] != r:
            hull.append(r)
        return hull

    points = sorted(points)
    l = functools.reduce(_keep_left, points, [])
    u = functools.reduce(_keep_left, reversed(points), [])
    return l.extend(u[i] for i in range(1, len(u) - 1)) or l


def primary(namef):

    p = os.path.join(namef)
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
    min_d = 0
    max_d = 0

    # img = np.ones((img_height, img_width), dtype='f8')*np.max(out_arr[:, 2])
    img = np.zeros((img_height, img_width), dtype='f8')
    for i in range(len(out_arr)):
        line = out_arr[i]
        d = line[2]
        row = int((img_height-1)*line[0])
        col = int((img_width-1)*line[1])
        # img[row, col] = max(d, img[row, col])
        img[row, col] = d
        # min_d = min(d, min_d)
        # max_d = max(d, max_d)

    # max_min_diff = max_d - min_d


    # def normalize(x):
    #     return 255- 255 * (x - min_d) / max_min_diff
    #
    # oblako = img.copy()
    # img = normalize(img)
    _fig, ax = plt.subplots()
    plt.imshow(img, cmap='gray_r')
    _fig.savefig('test.jpg')

    img = cv.imread('test.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)[70:420, 90:550]
    img = cv.blur(img, (5, 5))
    img = img.transpose()

    # namefile2d = path_to_2d+namef.replace('pcd','jpg')

    total = []
    # norm = mpl.colors.Normalize(vmin=-250, vmax=250)

    # img = cv.imread(namefile2d)
    # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)[70:420, 90:570]
    # img=cv.blur(img, (12,12))
    # img = img.transpose().astype(np.uint8)
    # plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='gray')
    plt.show()

    # return  3

    im1 = np.diff(img, axis=0, n=3)
    im2 = np.diff(img, axis=1, n=3)

    blank = np.zeros(im2.shape, dtype=np.uint8)
    cv.rectangle(blank,(50,400), (60,450), 255, -1)

    # point_pol = np.average(img[500:600, 50:100])
    point_pol = img[400, 50]
    blanck = np.zeros(img.shape, dtype=np.uint8)

    imgcopy = img.copy()
    # cv.rectangle(imgcopy,(0,0), (img.shape[0], 200),0,-1)
    mask_loc_max = cv.inRange(imgcopy,np.array([point_pol-5], dtype=int),np.array([point_pol+10], dtype=int))

    contours_niz, _ = cv.findContours(mask_loc_max, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    m=[]
    points = []
    for i, c in enumerate(contours_niz):
        cv.drawContours(blanck,contours_niz,i,1,1)
        if cv.contourArea(c)>S:
            m.append(i)


    for ind in m:
        cnt = contours_niz[ind]
        points += cnt.squeeze(axis=1).tolist()


    uncnt_peron = convex_hull_graham(points)
    peron_cnt = np.expand_dims(np.array(uncnt_peron), axis=1) # как контур

    cv.drawContours(blanck,[peron_cnt],0,1,1)

    plt.figure(figsize=(10,10))
    plt.title('перон')
    plt.imshow(blanck, cmap='gray_r')
    plt.show()

    # регрессия пола. относительно него будем измерять тела
    r_xy = []
    l_xy = []
    r_z = []
    l_z = []
    dist = []
    dist_arr = np.zeros((img.shape))


    nach_perona_top = min([x[1] for x in points])+10
    nach_perona_niz = max([x[1] for x in points])-10
    nach_perona_left = min([x[0] for x in points])+10
    nach_perona_right = max([x[0] for x in points])-10
    H,W = img.shape

    for y in range(nach_perona_top, nach_perona_niz, 10):
        for x in range(nach_perona_left, nach_perona_left + int(W * 0.1), 10):
            l_z.append(img[y, x])
            l_xy.append([x, y])
            # cv.circle(blanck,(x,y),1,1,1)

        for x in range(nach_perona_right-int(W * 0.1), nach_perona_right, 10):
            r_z.append(img[y, x])
            r_xy.append([x, y])
            # cv.circle(blanck,(x,y),1,1,1)

    linear_regression = LinearRegression()
    X = np.array(r_xy + l_xy)
    Y = np.array(r_z + l_z)
    linear_regression.fit(X, Y.reshape(-1, 1))

    # теперь в регрессии пол

    # на производной видны щели и окна

    contours, _ = cv.findContours(im1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blank = np.ones(im2.shape, dtype=np.uint8)
    blank2 = np.zeros(im2.shape, dtype=np.uint8)
    m = []
    for i, c in enumerate(contours):
        if cv.contourArea(c) > 1000:
            m.append(i)
    leftpoints = []
    rightpoints = []

    for ind in m:
        cv.drawContours(blank, contours, ind, 0, -1)


    contours, _ = cv.findContours(blank, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    m = []
    for i, c in enumerate(contours):
        if cv.contourArea(c) > 1000:
            m.append(i)

    # print(len(m))

    maxPoint = 0

    blank = np.zeros(im2.shape, dtype=np.uint8)
    for ind in m:
        cnt = contours[ind]
        maxPoint = max(maxPoint, max([line[1] for line in cnt.squeeze(axis=1).tolist()]))

    points = []
    for ind in m:
        cnt = contours[ind]
        nizc = max([line[1] for line in cnt.squeeze(axis=1).tolist()])
        if nizc > maxPoint - 50:
            cv.drawContours(blank2, contours, ind, 1, -1)
            points += cnt.squeeze(axis=1).tolist()

    uncnt_shel = convex_hull_graham(points)

    new_area = uncnt_peron + uncnt_shel

    min_niz = min(y for x, y in uncnt_shel)
    area = np.expand_dims(np.array(new_area), axis=1)  # как контур
    cv.drawContours(blank, [area], 0, 1, -1)
    plt.figure(figsize=(10, 10))
    plt.imshow(blank, cmap='gray')
    plt.figure(figsize=(10, 10))
    plt.imshow(blank2, cmap='gray')


    dist = []
    dist_arr = np.zeros((img.shape))

    for x in range(0, W, 1):
        for y in range(0, H, 1):
            if cv.pointPolygonTest(area, (x, y), False) == 1:
                p = linear_regression.predict(np.array([[x, y]]))[0, 0]
                delta = img[y, x]-p
                if 25<=delta<=125:
                # if -100<=delta<=-25:
                    dist.append(delta)
                    dist_arr[y, x] = delta

    # plt.figure(figsize=(10,10))
    # plt.hist(dist, 50, color='g')
    # plt.show()
    #
    # plt.figure(figsize=(10,10))
    # dist_arr = dist_arr[~np.all(dist_arr == 0, axis=1)]
    # dist_arr = dist_arr.transpose()
    # dist_arr = dist_arr[~np.all(dist_arr == 0, axis=1)]
    # dist_arr = dist_arr.transpose()
    # dist_arr[dist_arr == 0] = np.nan
    #
    # norm = mpl.colors.Normalize(vmin=-250, vmax=250)
    # plt.imshow(dist_arr, cmap='seismic', norm=norm)
    # plt.colorbar()
    # plt.title('Детекция объектов')
    # plt.show()

    blank = np.zeros(img.shape, dtype=np.uint8)

    contours, _ = cv.findContours(dist_arr.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        if cv.contourArea(c) > 100:
            cv.drawContours(blank,contours,i,1,-1)

            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            total.append(box)
            cv.drawContours(blank, [box], 0, 1, 2)


    plt.imshow(blank, cmap='gray')
    plt.title('Контуры объектов')
    plt.show()

    minx = np.min(out_arr[:,0])
    maxx = np.max(out_arr[:,0])
    miny = np.min(out_arr[:,1])
    maxy = np.max(out_arr[:,1])
    minz = np.min(out_arr[:,2])
    maxz = np.max(out_arr[:,2])

    center = []
    shirina = []
    for box in total:
        centrx = (box[0][0]+box[1][0]+box[2][0]+box[3][0])/4
        centry = (box[0][1]+box[1][1]+box[2][1]+box[3][1])/4
        center.append((centrx/img.shape[0] * (maxx - minx) + minx,
                       centry/img.shape[1] * (maxy - miny) + miny,
                       maxz - minz))
        minpox = min(box[0][0],box[1][0],box[2][0],box[3][0])
        maxpox = max(box[0][0],box[1][0],box[2][0],box[3][0])
        minpoy = min(box[0][1],box[1][1],box[2][1],box[3][1])
        maxpoy = max(box[0][1],box[1][1],box[2][1],box[3][1])
        shirinax = (maxpox-minpox)
        shirinay = (maxpoy-minpoy)
        shirina.append((shirinax/img.shape[0] * (maxx - minx) + minx,
                       shirinay/img.shape[1] * (maxy - miny) + miny,
                       1))


    return center, shirina

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--rgb", type=str,  default="../data/Сцена 4/0018.png", help="path to input rgb image")
    parser.add_argument("--depth", type=str,  default="../data/point_cloud_train/clouds_tof/cloud_0_0150.pcd", help="path to input depth image")
    args = parser.parse_args()
    bbox = primary(args.depth)
    print(bbox)