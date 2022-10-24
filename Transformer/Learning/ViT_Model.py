import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

transforms = transforms.Compose([
    transforms.ToTensor()
])

#step1 convert image to embedding vector sequence
def image2emb_native(image,patch_size,weight):
    # image shape:bs * channel * h * w
    patch=F.unfold(image,kernel_size=patch_size,stride=patch_size).transpose(-1,-2)
    patch_embedding=patch @ weight
    return patch_embedding

'''
    torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

    input: input tensor of shape (minibatch,in_channels,iH,iW)
    weight: filters of shape(out_channels,in_channels/groups,kH,kW)
    output: output tensor of shape (minibatch,out_channels,oH,oW)

'''
def image2emb_conv(image,kernel,stride):

    conv_output=F.conv2d(image,kernel,stride=stride)#bs * oc * oh *ow
    # Conv = nn.Conv2d(3, kernel.shape[0], 2, stride=2)
    # conv_output=Conv(image)#bs * oc * oh *ow
    bs,oc,oh,ow=conv_output.shape
    patch_embedding=conv_output.reshape((bs,oc,oh*ow)).transpose(-1,-2)
    return patch_embedding

# test code for image2emb
bs,ic,image_h,image_w=1,3,8,8
patch_size=4
model_dim=8
max_num_token=16
num_classes=10
patch_depth=patch_size * patch_size * ic
image=torch.randn(bs,ic,image_h,image_w)
label=torch.randint(10,(bs,))
# image=transforms(Image.open('./cat.png').resize((16,16)))
# image=torch.unsqueeze(image,0)
weight=torch.randn(patch_depth,model_dim)#model_dim是输出通道数目，patch_depth是卷积核的面积乘以输入通道数

# patch_embedding_naive=image2emb_native(image,patch_size,weight)
kernel=weight.transpose(0,1).reshape((-1,ic,patch_size,patch_size))#oc*ic*kh*kw

patch_embedding_conv=image2emb_conv(image,kernel,patch_size)
# print(patch_embedding_naive.shape)
print(patch_embedding_conv.shape)

#step2 prepend CLS token embedding
cls_token_embedding=torch.randn(bs,1,model_dim,requires_grad=True)
token_embedding=torch.cat([cls_token_embedding,patch_embedding_conv],dim=1)

#step3 add position embedding
position_embedding_table=torch.randn(max_num_token,model_dim,requires_grad=True)
seq_len=token_embedding.shape[1]
position_embedding=torch.tile(position_embedding_table[:seq_len],[token_embedding.shape[0],1,1])
token_embedding += position_embedding

#step4 pass embedding to Transformer Encoder
encoder_layer=nn.TransformerEncoderLayer(d_model=model_dim,nhead=8)
transform_encoder=nn.TransformerEncoder(encoder_layer,num_layers=6)
encoder_output=transform_encoder(token_embedding)

#step5 do classification
cls_token_output=encoder_output[:,0,:]
linear_layer=nn.Linear(model_dim,num_classes)
logits=linear_layer(cls_token_output)
loss_fn=nn.CrossEntropyLoss()
loss=loss_fn(logits,label)
print(loss)

