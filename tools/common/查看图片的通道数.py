#查看图片的通道数
from PIL import Image
c=Image.open('E:/python/cnn_demo/data_list/train/000.png')
print(len(c.split()))