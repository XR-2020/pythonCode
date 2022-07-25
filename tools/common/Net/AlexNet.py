from torch import nn
import torch
# from Alexnet.AlexNet import *
from torch.autograd import Variable

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc = nn.Sequential(
            #in_features由输入张量的形状决定，out_features则决定了输出张量的形状
            nn.Linear(in_features = 256, out_features = 1)
        )

    def forward(self, x):
        x = self.conv(x)
        #x.view(1,256)将四维张量转换为二维张量之后，才能作为全连接层的输入
        y=self.fc(x.view(1,256))
        print(x.size())
        print(y.size())


if __name__ == '__main__':
  net = AlexNet()

  data_input = Variable(torch.randn([1, 3, 96, 96])) # 这里假设输入图片是96x96
  print(data_input.size())
  net(data_input)