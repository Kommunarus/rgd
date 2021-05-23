
import os
from run_one_img import primary

dir = '/home/neptun/Документы/Хакатон/point_end/clouds_stereo/'

listf = os.listdir(dir)

for f in listf:
    path = os.path.join(dir, f)
    primary(path_depth=path, usegpu=False, PATH='./rjd_3d_best_3.pth')