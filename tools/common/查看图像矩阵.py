# 查看图像矩阵
import json
import numpy as np
from PIL import Image

f = open("out.json", "w")  # 打开写入图像矩阵信息的文件
c = Image.open("/tools/points_to_binary_images/test.png")  # 打开文件
f.write(json.dumps(np.array(c).tolist()))  # 将图像矩阵写入文件
f.close()  # 关闭文件

