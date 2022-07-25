#绘画图像

from PIL import Image
import matplotlib.pyplot as plt

# 图片路径
img = Image.open("E:/python/tools/common/P02-0148.png").convert('L')
plt.figure("Image")  # 图像窗口名称
plt.imshow(img,cmap='gray')
plt.axis('on')  # 关掉坐标轴为 off
plt.title('image')  # 图像题目
plt.grid(True,linewidth=1)
plt.plot(120.397032,111.786940, '.', markersize=1, color='r')
plt.plot(163.563163,104.087621, '.', markersize=1, color='g')
# 必须有这个，要不然无法显示
plt.show()