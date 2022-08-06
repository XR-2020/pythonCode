import os

import cv2 as cv
import numpy as np

all_dir='D:/graduate_study/test2/'
image_dir='D:/full_data/result/test2/icontour'
img_list=os.listdir(image_dir)
for item in img_list:
    dir_path = os.path.join(all_dir, item[0:3] + 'contours-auto')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        # 创建图片的txt文件
    file = open(dir_path + '/' + item[0:8] + '-icontour-auto.txt', mode='w', encoding='utf-8')
    # 读取图像
    img_source = cv.imread(os.path.join(image_dir, item))  # 原图（三通道）
    # 将图像转为灰度图
    img = cv.cvtColor(img_source, cv.COLOR_BGR2GRAY)
    # 检测轮廓
    contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for contour in range(len(contours)):
        # 转换数据形式为（x,y）
        flat_list = np.ravel(contours[contour]).tolist()
        list_of_tuples = list(zip(flat_list[0::2], flat_list[1::2]))
        for points in range(len(list_of_tuples)):
            writedata = str(list_of_tuples[points]).replace('(', '').replace(')', '\n').replace(',', ' ')
            file.write(writedata)
    file.close()