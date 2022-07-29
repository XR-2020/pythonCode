import torch
import cv2
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torchsummary import summary

class Mydataset(Dataset):
    def __init__(self, file_path):
        self.data = []#图像数据，list
        self.target = []#标签数据,list
        data_y = []#存放图像标签,list
        data_x = []#存放图像,list
        image_dir = os.path.join(file_path)#图像文件夹地址,str
        list_image = os.listdir(image_dir)#图像文件列表,list
        for image in list_image:
            if 'mask' in image:
                data_y_path = os.path.join(image_dir, image)
                datay = cv2.imread(data_y_path, cv2.IMREAD_GRAYSCALE)#以灰度图像模式读入图像ndarray类型
                datay = cv2.resize(datay, (256, 256))#将图片尺寸改为256*256大小
                """
                datay/255：
                    将像素值归一化，使像素值在0-1之间,
                    1.减小数字大小,降低计算难度
                    2.防止通吃效果即像素值过大会对最后结果造成很大影响，不同像素值对结果影响程度不同
                torch.Tensor（）
                    将图像的数据类型由ndarray转为3维Tensor
                """
                datay = torch.Tensor(datay/255)#DataLoader中的dataset要求输入为Tensor，在此之前datay均为ndarray类型
                datay = datay.view(1, 256, 256)#重构图像
                data_y.append(image,datay)
            else:
                data_x_path = os.path.join(image_dir, image)
                datax = cv2.imread(data_x_path, cv2.IMREAD_GRAYSCALE)#ndarray
                datax = cv2.resize(datax, (256, 256))
                datax = torch.Tensor(datax/255)
                datax = datax.view(1, 256, 256)
                data_x.append(image,datax)
        #堆叠使3维Tensor转为4维Tensor
        self.data = torch.stack(data_x)#Tensor,4维
        self.target = torch.stack(data_y)#Tensor
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.target[index, :, :, :]

    def __len__(self):
        return self.len


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = self.conv_sequential(1, 16, 3, 1, 2, 2)  # 1表示通道数，16个filter,每个小是3*3，padding为1，maxpooling
        # 的filter为2*2，stride=2
        self.layer2 = self.conv_sequential(16, 64, 3, 1, 2, 2)
        self.layer3 = self.conv_sequential(64, 128, 3, 1, 2, 2)
        self.layer4 = self.conv_sequential(128, 256, 3, 1, 2, 2)
        self.layer5 = self.conv_sequential(256, 512, 3, 1, 2, 2)
        self.transpose_layer2 = self.transpose_conv_sequential(1, 1, 4, 2, 1)#转置卷积
        self.transpose_layer8 = self.transpose_conv_sequential(1, 1, 16, 8, 4)
        self.ravel_layer32 = nn.Sequential(
            nn.Conv2d(512, 1, 1),  # 1个filter,每个大小是1*1
            nn.ReLU(True)
        )
        self.ravel_layer16 = nn.Sequential(
            nn.Conv2d(256, 1, 1),  # 1个filter,每个大小是1*1
            nn.ReLU(True)
        )
        self.ravel_layer8 = nn.Sequential(
            nn.Conv2d(128, 1, 1),  # 1个filter,每个大小是1*1
            nn.ReLU(True)
        )

    def forward(self, x):
        list=[]
        ret = self.layer1(x)  # 此时网络应该输出16个大小为64*64的特征图
        list.append('layer1',ret)
        ret = self.layer2(ret)  # 此时网络应该输出64个大小为32*32的特征图
        list.append('layer2',ret)
        ret = self.layer3(ret)  # 此时网络应该输出128个大小为16*16的特征图
        list.append('layer3',ret)
        x8 = ret
        ret = self.layer4(ret)  # 此时网络应该输出256个大小为8*8的特征图
        list.append('layer4',ret)
        x16 = ret
        ret = self.layer5(ret)  # 此时网络应该输出512个大小为4*4的特征图
        list.append('layer5',ret)
        x32 = ret
        x32 = self.ravel_layer32(x32)  # 此时网络应该输出1个大小为4*4的特征图
        list.append('ravel_layer32',x32)
        x16 = self.ravel_layer16(x16)  # 应该输出1个大小为8*8的特征图
        list.append('ravel_layer16',x16)
        x8 = self.ravel_layer8(x8)  # 应该输出1个大小为16*16的特征图
        list.append('ravel_layer8',x8)
        x32 = self.transpose_layer2(x32)  # 应该输出1个大小为8*8的特征图
        list.append('x32transpose_layer2',x32)
        x16 = x16 + x32
        list.append('x16 + x32',x16)
        x16 = self.transpose_layer2(x16)  # 应该输出1个大小为16*16的特征图
        list.append('x16transpose_layer2',x16)
        x8 = x8 + x16
        list.append('x8 + x16', x8)
        result = self.transpose_layer8(x8)  # 应该输出1个大小为128*128的特征图
        list.append('transpose_layer8',result)
        out = nn.Sigmoid()(result)
        return out,list

    def conv_sequential(self, in_size, out_size, kfilter, padding, kernel_size, stride):
        """
        Args:
            in_size: 输入通道个数
            out_size: 输出通道个数
            kfilter: 卷积时卷积核大小
            padding: 填充
            kernel_size:池化时卷积核大小
            stride: 池化步长
        Returns:

        """
        model=nn.Sequential(
            nn.Conv2d(in_size, out_size, kfilter, padding=padding),

            nn.BatchNorm2d(out_size),
            # 1.当inplace=True的时候，会改变输入数据；当inplace=False的时候，不会改变输入数据，
            # 2.激活函数：ReLU（）将小于0的值置为0，大于0 的保存下来
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size, stride)
        )
        return model
    #逆卷积
    def transpose_conv_sequential(self, in_size, out_size, kfilter, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kfilter, stride, padding, bias=False),
            nn.BatchNorm2d(out_size)
        )


def train():
    batch_size = 4
    trainfile_path = './data_list/train'
    """
    traindataset = Mydataset(trainfile_path)：
        执行完成后traindataset数据类型为Mydataset，包含data(Tensor,4维)，len(int),target(Tensor,4维)
    """
    traindataset = Mydataset(trainfile_path)
    #DataLoader中的dataset要求输入为Tensor
    train_dataloader = DataLoader(
        dataset=traindataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    # model = UNet(1, 2)
    model = Net()
    # 查看每层的输出大小
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    summary(model, input_size=(1, 256, 256))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_function = nn.BCELoss()
    num_size = 1
    for num in range(num_size):
        Loss = 0
        print('----第%d轮迭代-----' % num)
        """
             for i, data in enumerate(train_dataloader):
                i（list）：一张图片在网络中各层的Tensor,每一层一个文件
                data（list）:一张输入图片的Tensor,一张目标标签图片的Tensor,一张图片一个文件夹,原图是第一个文件，标签是第二个文件
        """
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            prediction,fm = model(inputs)
            save_featureMap(fm)
            loss = loss_function(prediction, labels)
            Loss += loss
            loss.backward()
            optimizer.step()
        print('损失为%f' % Loss.item())
    return model, optimizer


def test(model):
    testfile_path = './data_list/test'
    testdataset = Mydataset(testfile_path)
    test_dataloaders = DataLoader(dataset=testdataset, batch_size=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for i, data in enumerate(test_dataloaders):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            prediction,i = model(inputs)
            img_y = torch.reshape(prediction, (256, 256)).detach().cpu().numpy()  # 取出数值部分，原大小是1*1*128*128
            # 对应的二值图
            img = np.round(img_y)  # 预测标签
            img = img * 255  # *255
            im = Image.fromarray(img)  # numpy 转 image类
            im = np.array(im, dtype='uint8')
            Image.fromarray(im, 'L').save("%03d.png" % i)
            i = i + 1
            plt.pause(0.01)
"""
    list:每张图片训练完一次后的特征图数据列表
    list([1,256,256,16],[1,128,128,16],[1,128,128,64],[1,64,64,64],...........)
"""
def save_featureMap(list):
    for a, feature_map in enumerate(list):  # a（int）：list中的元素下标，feature_map:元素本身
        # [N, C, H, W] -> [C, H, W]降维并转为ndarray
        im = np.squeeze(feature_map.detach().numpy())
        # [C, H, W] -> [H, W, C]
        # 需要改变才可以正常显示图像，调整通道顺序
        im = np.transpose(im, [1, 2, 0])
        # 设置大小
        plt.figure(figsize=(feature_map.shape[2] / 100, feature_map.shape[3] / 100), dpi=100)
        plt.axes([0., 0., 1., 1.])
        for i in range(feature_map.shape[1]):  # 列表中一项里每个特征图生成
            # ax = plt.subplot(4, 4, i + 1)
            # [H, W, C]  cmap='gray' :设置为灰度图， [:, :, i]选择对channels进行切分
            plt.imshow(im[:, :, i], cmap='gray')
            plt.axis('off')
            # 保存图像
            plt.savefig('picture/featureMap{}_{}_outputs.png'.format(feature_map.shape[1], i), dpi=100,
                        bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':

    net, optimizer= train()

    # plt.imsave(batchidx, arr, format='jpg')

        # plt.show()

    # # 模型保存
    # torch.save(net, 'cnn.pth')
    #
    # model = torch.load("cnn.pth")
    # test(model)
    # print('done')
