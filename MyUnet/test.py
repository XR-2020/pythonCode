import os

import torch

from MyUnet.utils import keep_image_size_open
from net import *
from data import *
from torchvision.utils import save_image
from torchvision import transforms


transform = transforms.Compose([transforms.ToTensor()])
net = UNet().cpu()

weights = 'params/unet.pth'
test_path='E:/python/MyUnet/test_data/image/'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')
#测试数据文件夹中所有数据
#获取所有文件名
filelist = os.listdir(test_path)
for item in filelist:
    print(os.path.join(test_path,item))
    img = keep_image_size_open(os.path.join(test_path,item))
    img_data=transform(img).cpu()
    """
    torch.unsqueeze(tensor,dim,out)
        tensor (Tensor) – 输入张量
        dim (int) – 插入维度的索引
        out (Tensor, optional) – 结果张量
    原张量与改变后的张量共存改变一个则都改变
    """
    img_data=torch.unsqueeze(img_data, dim=0)#对数据维度进行扩充
    out = net(img_data)
    save_image(out, f'E:/python/MyUnet/test_data/label/{item}')
