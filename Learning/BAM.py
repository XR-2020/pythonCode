import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import cv2 as cv
from torchvision import transforms

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256))
])
class BAM(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in //8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        #print('ds: ',ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, input):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        #1，256，8，8
        x = self.pool(input)
        #print("1",x.shape)

        #1，256，8，8
        m_batchsize, C, width, height = x.size()


        #将1，256，8，8 变为 1，64，32
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        #print("2",proj_query.shape)

        # 将1，256，8,8 变为 1，32，64
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        #print(proj_key.shape)

        #1，64，32x1，32，64----1，64，64
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        #print(energy.shape)

        #计算A
        energy = (self.key_channel**-.5) * energy
        #print("3",energy.shape)

        #经过softmax得到相似矩阵A
        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)
        #print(attention.shape)

        #reshape V  256x64
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        #print(proj_value.shape)

        #256x64 x  64x64  得到 256 64
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        #reshape为256 x8x8
        out = out.view(m_batchsize, C, width, height)

        #经过双线性插值恢复为原图大小
        out = F.interpolate(out, [width*self.ds,height*self.ds])
        out = out + input
        return out

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(3,128,3,1,padding=1),
        )
    def forward(self,x):
        return self.conv(x)

class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(128,3,3,1,padding=1),
        )
    def forward(self,x):
        return self.conv(x)

if __name__ == '__main__':
    img=cv.imread('img_1.png')
    img=transforms(img)
    img=img.unsqueeze(0)

    for i in range(3):
        conv = Conv()
        x = conv(img)

        model = BAM(128)
        out = model(x)

        conv2 = Conv2()
        img = conv2(out)


    show=np.squeeze(img.detach().numpy(),0)
    show=np.transpose(show, [1, 2, 0])
    cv.imshow('out',show)
    cv.waitKey()
    cv.destoryAllwindows()
