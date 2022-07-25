import torch
from torch import nn
from torch.nn import functional as F

#实现UNet网络
class Conv_Block(nn.Module):#卷积层
    def __init__(self, in_channel, out_channel):
        """
             nn.Conv2d(in_channel, out_channel     in_channel由自己指定，out_channel=输入通道数+输入通道数=2*in_channel
                in_channel, out_channel指的是filter输入通道数，输出通道数
        """
        super(Conv_Block, self).__init__()
        """
            BatchNorm2d()
                神经网络对0附近的数据更敏感, 但是随着网络层数的增加，特征数据会出现偏离0均值的情况,使用标准化，将偏移的数据重新拉回
                批标准化：是对一个batch的数据做标准化处理，常用在卷积操作和激活操作之间
        """
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),#第一次卷积，卷积核3*3，padding为1，步长为1
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),#舍弃率
            nn.LeakyReLU(),#激活
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),#第二次卷积
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),#下采样
            nn.LeakyReLU()#激活函数
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):#下采样的实现
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),#卷积核3*3，padding为2，步长为1
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()#激活函数
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):#上采样
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)#通道每次降为原来一半，卷积核1，步长1

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)

class UNet (nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.c1=Conv_Block(3,64)
        self.d1=DownSample(64)#特征提取第一层结束
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)#特征提取第二层结束
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)#特征提取第三层结束
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)#特征提取第四层结束
        self.c5 = Conv_Block(512, 1024)#特征提取最后一次卷积
        self.u1 = UpSample(1024)#特征还原第一次上采样
        self.c6=Conv_Block(1024,512)
        self.u2=UpSample(512)#特征还原第一层结束
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)#特征还原第二层结束
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)#特征还原第三层结束
        self.c9 = Conv_Block(128, 64)#特征还原第四层结束
        # 输出3通道图片，进行全连接层(1,64,256,256)->(1,3,256,256)
        self.out=nn.Conv2d(64,3,3,1,1)
        self.Th=nn.Sigmoid()


    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3=self.c3(self.d2(R2))
        R4=self.c4(self.d3(R3))
        R5=self.c5(self.d4(R4))
        o1=self.c6(self.u1(R5,R4))
        o2=self.c7(self.u2(o1,R3))
        o3=self.c8(self.u3(o2,R2))
        o4=self.c9(self.u4(o3,R1))

        return self.Th(self.out(o4))


if __name__ == '__main__':
    x=torch.randn(2,3,256,256)
    net=UNet()
    print(net(x).shape)




