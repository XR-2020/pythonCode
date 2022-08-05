#改变图片尺寸
#提取目录下所有图片,更改尺寸后保存到另一目录
from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=216,height=256):
    img2=Image.open(jpgfile)
    img=img2.copy()
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


if __name__=='__main__':
    path="./"
    list=os.listdir(path)
for jpgfile in list:
    if '.png' in jpgfile:
        convertjpg(os.path.join(path,jpgfile),"./")
