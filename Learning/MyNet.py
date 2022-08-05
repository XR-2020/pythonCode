import os

import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms,datasets
from torch.nn import functional as F
from cnn_demo.cnnSeg_torch import summary

transforms = transforms.Compose([
    transforms.ToTensor()
])
def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    """
        Image.new(mode, size, color=0)
            mode:模式，通常用"RGB"这种模式，如果需要采用其他格式，可以参考博文：PIL的mode参数
            size：生成的图像大小
            color：生成图像的颜色，默认为0，即黑色
    """
    mask = Image.new('L', (temp, temp), 0)
    # 将一张图粘贴到另一张图像上,paste(image,box)变量box是一个给定左上角坐标的2元组
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'label'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png
        segment_path = os.path.join(self.path, 'label', segment_name)#拼接出训练数据的标签地址
        image_path = os.path.join(self.path, 'image', segment_name)#拼接出训练数据的原图地址
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transforms(image), transforms(segment_image)

data_path='E:/python/Learning/picture/'
data_loader=DataLoader(MyDataset(data_path),batch_size=1,shuffle=False)
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.inlayer=nn.Sequential(
            nn.Conv2d(1,64,3,1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(64),
            # nn.LeakyReLU()
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(64),
            # nn.LeakyReLU()
            nn.ReLU(),
            nn.MaxPool2d(2),

            #self.conv_dw(64, 128, 1),
            nn.Conv2d(64, 128, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(128),
            # nn.LeakyReLU()
            nn.ReLU(),

            #self.conv_dw(128, 128, 1),
            nn.Conv2d(128, 128, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(128),
            # nn.LeakyReLU()
            nn.ReLU(),
            nn.MaxPool2d(2),

            #self.conv_dw(128, 256, 1),
            nn.Conv2d(128, 256, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(256),
            # nn.LeakyReLU()
            nn.ReLU(),

            #self.conv_dw(256, 256, 1),
            nn.Conv2d(256, 256, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(256),
            # nn.LeakyReLU()
            nn.ReLU(),
            nn.MaxPool2d(2),

            #self.conv_dw(256, 512, 1),
            nn.Conv2d(256, 512, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(512),
            # nn.LeakyReLU()
            nn.ReLU(),

            #self.conv_dw(512, 512, 1),
            nn.Conv2d(512, 512, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(512),
            # nn.LeakyReLU()
            nn.ReLU(),
            nn.MaxPool2d(2),

            #self.conv_dw(512, 1024, 1),
            nn.Conv2d(512, 1024, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(1024),
            # nn.LeakyReLU()
            nn.ReLU(),
            #self.conv_dw(1024, 1024, 1),
            nn.Conv2d(1024, 1024, 3, 1,1, padding_mode='reflect', bias=False)
        )
        self.outlayer1=nn.Sequential(
            nn.Conv2d(1024,512,3,1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(512),
            # nn.LeakyReLU()
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(512),
            # nn.LeakyReLU()
            nn.ReLU()
        )
        self.outlayer2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(256),
            # nn.LeakyReLU()
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(256),
            # nn.LeakyReLU()
            nn.ReLU()
        )
        self.outlayer3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(128),
            # nn.LeakyReLU()
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(128),
            # nn.LeakyReLU()
            nn.ReLU()
        )
        self.outlayer4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(64),
            # nn.LeakyReLU()
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1,1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(64),
            #nn.LeakyReLU()
            nn.ReLU()
        )
        self.last = nn.Sequential(
            nn.Conv2d(64, 1, 1, 1),
            nn.Sigmoid()
        )
    def forward(self,input):
        input=self.inlayer(input)
        input=F.interpolate(input,scale_factor=2,mode='nearest')
        input=self.outlayer1(input)
        input=F.interpolate(input,scale_factor=2,mode='nearest')#上采样
        input=self.outlayer2(input)
        input = F.interpolate(input, scale_factor=2, mode='nearest')
        input = self.outlayer3(input)
        input = F.interpolate(input, scale_factor=2, mode='nearest')
        input = self.outlayer4(input)
        input = self.last(input)
        return input
    def conv_dw(self,inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp,bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(inp),
            nn.ReLU(),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(),
        )

if __name__ == '__main__':
    save_path='./picture'
    net=MyNet()
    summary(net, input_size=(1, 256, 256))
    loss_fn=nn.BCELoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=0.01)

    total=130
    loss_total=0
    point_x=[]
    point_y=[]
    for epoch in range (total):
        loss_total = 0
        print("**********{}".format(epoch))
        for index, (image, label) in enumerate(data_loader):
            output=net(image)
            loss=loss_fn(output,label)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(epoch % 10==0):
                _out_image = output[0]
                save_image(_out_image, f'{save_path}/{epoch}.png')
            torch.save(net, 'params.pth')
        plt.scatter(epoch, loss.item(), s=8,color='blue')
        point_x.append(epoch)
        point_y.append(loss.item())
    plt.plot(point_x, point_y, linewidth=1, color='b')
    plt.title('loss', fontsize=20)
    plt.show()


test_path='img.png'
test_image=keep_image_size_open(test_path)
test=transforms(test_image)
model=torch.load('params.pth')
test=torch.unsqueeze(test, dim=0)
outputs=model(test)
print(outputs.shape)
save_image(outputs, 'test.png')

# import os
#
# import torch
# import torchvision
# from PIL import Image
# from torch import nn
# from torch.utils.data.txt import DataLoader
# from torchvision.utils import save_image
# from torchvision.datasets import ImageFolder
# from torchvision import transforms,datasets
# from torch.nn import functional as F
#
# transforms = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])
#
# data_images_path ='E:/python/Learning/picture/image/61a.png'
# label_images_path ='E:/python/Learning/picture/label/61a.png'
# data_images=Image.open(data_images_path)
# test_images=Image.open(label_images_path)
# data_loader=DataLoader(transforms(data_images),batch_size=1,shuffle=False)#初始化数据集
# label_loader=DataLoader(transforms(test_images),batch_size=1,shuffle=False)
# for i,(image,target) in enumerate (data_loader):
#     print(image.shape)
# print()
# class MyNet(nn.Module):
#     def __init__(self):
#         super(MyNet, self).__init__()
#         self.inlayer=nn.Sequential(
#             nn.Conv2d(3,64,3,1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(64, 128, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(128, 256, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(256, 512, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(512, 1024, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.Conv2d(1024, 1024, 3, 1,1, padding_mode='reflect', bias=False)
#         )
#         self.outlayer1=nn.Sequential(
#             nn.Conv2d(1024,512,3,1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.Conv2d(512,512,3,1,1, padding_mode='reflect', bias=False),
#             nn.ReLU()
#         )
#         self.outlayer2 = nn.Sequential(
#             nn.Conv2d(512, 256, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU()
#         )
#         self.outlayer3 = nn.Sequential(
#             nn.Conv2d(256, 128, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU()
#         )
#         self.outlayer4 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, 1,1, padding_mode='reflect', bias=False),
#             nn.ReLU()
#         )
#         self.last = nn.Conv2d(64,3,1,1)
#     def forward(self,input):
#         input=self.inlayer(input)
#         input=F.interpolate(input,scale_factor=2,mode='nearest')
#         input=self.outlayer1(input)
#         input=F.interpolate(input,scale_factor=2,mode='nearest')
#         input=self.outlayer2(input)
#         input = F.interpolate(input, scale_factor=2, mode='nearest')
#         input = self.outlayer3(input)
#         input = F.interpolate(input, scale_factor=2, mode='nearest')
#         input = self.outlayer4(input)
#         input = self.last(input)
#         return input
#
# if __name__ == '__main__':
#     net=MyNet()
#     loss_fn=nn.BCELoss()
#     optimizer=torch.optim.Adam(net.parameters(),lr=0.01)
#
#     total=200
#
#     for epoch in range (total):
#             output=net(data_loader)
#             loss=loss_fn(output,label_loader)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#     torch.save(net,'params.pth')
#
# test_path='img.png'
# test_image=Image.open(test_path).convert('RGB')
# test=transforms(test_image)
# model=torch.load('params.pth')
# test=torch.reshape(test,(1,3,256,256))
# outputs=model(test)
# print(outputs.shape)
# save_image(outputs, 'test.png')