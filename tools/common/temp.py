import cv2 as cv
import numpy as np

image_path='E:/python/tools/common/P17-0080.png'
# 读取图像
img_source = cv.imread(image_path) # 原图（三通道）
# 将图像转为灰度图
img = cv.cvtColor(img_source, cv.COLOR_BGR2GRAY).astype(np.float32)/255.0
# 检测轮廓
contours, hierarchy = cv.findContours(img,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# contours=contours.astype(np.float32)/255.0
# cv.drawContours(test_source, contours, -1,(0, 0, 0),2)
# cv.imshow("result", test_source)
# cv.waitKey()
flat_list = np.ravel(contours[0]).tolist()
list_of_tuples = list(zip(flat_list[0::2], flat_list[1::2]))
file = open('point.txt', mode='w',encoding='utf-8')
writedata = str(list_of_tuples[0]).replace('(','').replace(')','\n').replace(',',' ')
writedata1 = str(list_of_tuples[1]).replace('(','').replace(')','\n').replace(',',' ')
file.write(writedata)
file.write(writedata1)



"""
创建纯白图片
"""

# import numpy as np
# import matplotlib.pyplot as plt
# img=np.zeros((216,256,1),np.uint8)
# plt.figure(figsize=(216 / 100, 256 / 100), dpi=100)
# plt.axes([0., 0., 1., 1.])
# plt.axis('off')
# plt.savefig('test.png', dpi=100, bbox_inches='tight', pad_inches=0)
