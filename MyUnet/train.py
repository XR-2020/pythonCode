from  torch import nn,optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path='params/unet.pth'
data_path=r'E:/python/MyUnet/train_data/image/'
save_path='train_image'#可视化训练过程
if __name__ == '__main__':
    data_loader=DataLoader(MyDataset(data_path),batch_size=1,shuffle=True)#初始化数据集,shuffle=True是否打乱顺序
    net=UNet().to(device)#将神经网络放在设备上
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successfully load weight')
    else:
        print('not successful load weight')
    opt=optim.Adam(net.parameters())#pytorch 优化器(optim)
    loss_fun=nn.BCELoss()#二分类交叉熵损失

    epoch=1
    while True:
        for i,(image,segment_image) in enumerate(data_loader):
            image,segment_image=image.to(device),segment_image.to(device)#将标签和图片放在设备上
            out_image=net(image)#得出训练结果
            train_loss=loss_fun(out_image,segment_image)#计算损失
            opt.zero_grad()## 清空之前保留的梯度信息
            train_loss.backward()# 将BCELoss的loss 信息反传回去
            opt.step()#根据 optim参数 和 梯度 更新参数


            print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            if i%30==0:
                torch.save(net.state_dict(),weight_path)

            _image=image[0]
            _segment_image=segment_image[0]
            _out_image=out_image[0]

            img=torch.stack([_image,_segment_image,_out_image],dim=0)
            save_image(img,f'{save_path}/{i}.png')

        epoch+=1
