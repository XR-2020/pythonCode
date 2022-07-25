import cv2 as cv
from matplotlib import  pyplot as plt

"""
    查看图像的灰度直方图
"""

img=cv.imread('last.png')
plt.hist(img.ravel(),256,[0,256])
plt.show()