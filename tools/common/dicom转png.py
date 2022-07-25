import pydicom
import matplotlib.pyplot as plt
import scipy.misc
import pandas as pd
import numpy as np
import os
import pydicom
import matplotlib.pyplot as plt
from matplotlib import image
import scipy.misc
from PIL import Image

#dicom转png并且转换位深

# dicom文件转为png图片
def Dcm2jpg(file_path, outpath):
    # 拿到文件目录
    list = os.listdir(file_path)
    # 遍历每个文件
    for item in list:
        # 拼接出文件地址
        path = os.path.join(file_path, item)
        # 读取文件数据
        filedata = pydicom.read_file(path)
        # 提取图像信息
        img = filedata.pixel_array
        #拼出图像名称
        imgname=item.split('.')[0] + '.png'
        # 拼接出图像路径
        imgpath = os.path.join(outpath,imgname)
        # 保存为灰度图像
        image.imsave(imgpath, img, cmap='gray')
    print("OK")

#改变图片位深
def changebit(path):
    imagelist=os.listdir(path)
    for image in imagelist:
        image_path = os.path.join(path, image)
        img = Image.open(image_path)  # 打开图片
        # print(img.format, img.size, img.mode)  # 打印出原图格式
        img = img.convert("L")  # 4通道转化为灰度L通道
        img.save(os.path.join(path, image))
    print("转换完成")


if __name__ == '__main__':
    dicompath='E:/graduate_study/work/full_dataset/Test2SetContours_dicoms'
    pngpath='E:/graduate_study/work/full_dataset/Test2SetContours_images'
    Dcm2jpg(dicompath,pngpath)
    changebit(pngpath)
