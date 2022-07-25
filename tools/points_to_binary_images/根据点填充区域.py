import cv2 as cv
import numpy as np
from PIL import Image


#读取图像
img_source = cv.imread("test.png")  #原图（三通道）
#将图像转为黑白图
img = cv.cvtColor(img_source, cv.COLOR_BGR2GRAY)
#检测轮廓
contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# 给轮廓填充颜色**轮廓一定要闭合
cv.fillPoly(img_source, [contours[1]], (255, 255, 255))  # 填充内部
# cv.imshow("result", img_source)
#转为灰度图
save_image=img = cv.cvtColor(img_source, cv.COLOR_BGR2GRAY)
#保存图片
cv.imwrite('result.png',save_image)
#销毁窗口
cv.destroyAllWindows()

