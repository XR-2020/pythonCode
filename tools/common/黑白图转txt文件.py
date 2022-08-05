import cv2 as cv
import numpy as np

image_path='E:/python/tools/common/P17-0080.png'
# 读取图像
img_source = cv.imread(image_path) # 原图（三通道）
# 将图像转为灰度图
img = cv.cvtColor(img_source, cv.COLOR_BGR2GRAY)
# 检测轮廓
contours, hierarchy = cv.findContours(img,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#创建图片的txt文件
file = open('P17-0080-ocontour-auto.txt', mode='w',encoding='utf-8')
for contour in range(len(contours)):
    # 转换数据形式为（x,y）
    flat_list = np.ravel(contours[contour]).tolist()
    list_of_tuples = list(zip(flat_list[0::2], flat_list[1::2]))
    for points in range(len(list_of_tuples)):
        writedata = str(list_of_tuples[points]).replace('(','').replace(')','\n').replace(',',' ')
        file.write(writedata)