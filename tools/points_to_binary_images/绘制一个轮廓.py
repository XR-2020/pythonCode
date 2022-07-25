import os

from PIL import Image
from pylab import *
# import matplotlib.pyplot as plt
import cv2
from matplotlib import pyplot as plt

"""
    绘制一张图片的轮廓
"""

#拿到存放标注文件的文件夹路径
filepath= '/tools/points_to_binary_images/P01-0080-icontour-manual.txt'
#拿到放图片的文件夹的路径
imagepath='E:/python/tools/points_to_binary_images/ref.png'
#标注图像的存放位置
savepath='E:/python/tools/points_to_binary_images'
#打开内轮廓标注文件
iconfile = open(filepath)
#打开被标注图片
img = array(imread(imagepath))
#设置以灰度图片方式打开
image_open = imread(imagepath)
img = array(image_open)
# 设置大小
plt.figure(figsize=(image_open.shape[1] / 100, image_open.shape[0] / 100), dpi=100)
plt.axes([0., 0., 1., 1.])
plt.imshow(img, cmap='gray')
#读取文件每一行数据
icon_sourceInLine = iconfile.readlines()#内轮廓
icon_pointspos = []  # 内轮廓点
#处理内轮廓点数据
for icon_line in icon_sourceInLine:
    temp1 = icon_line.strip('\n')
    temp2 = temp1.split()
    icon_pointspos.append(temp2)
for icon in range(0, len(icon_pointspos)):
    icon_x = float(icon_pointspos[icon][0])
    icon_y = float(icon_pointspos[icon][1])
    plt.plot(icon_x, icon_y, '.', markersize=1, color='w')
#关闭坐标轴
plt.axis('off')
#拼接出标注图的位置
save_image_path=os.path.join(savepath,'test.png')
#保存图片,除白边
plt.savefig(save_image_path, dpi=100, bbox_inches='tight', pad_inches=0)
plt.clf()
