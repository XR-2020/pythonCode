import os

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transforms = transforms.Compose([
    transforms.ToTensor()
])

#构建UNet的数据集
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


if __name__ == '__main__':
    data = MyDataset('E:\python\MyUnet\data')
    #[0][0]图片
    print(data[0][0].shape)
    #[0][1]标签
    print(data[0][1].shape)
