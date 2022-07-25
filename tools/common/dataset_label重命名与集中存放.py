import os

#labelme的dataset中label重命名与集中存放
class BatchRename():

    def rename(self):
        path = "E:/graduate_study/work/labelme/b/"
        filelist = os.listdir(path)
        for item in filelist:  # 获取所有dataset名称
            nextlist = os.listdir(path + item)  # 获取每个dataset文件夹下文件的名字
            for img in nextlist:
                if 'label' in img and img.endswith('.png') and len(img) == 9:  # 找出标签图片
                    src = os.path.join(os.path.abspath(path), item, img)#拼接标签图片路径
                    dst = os.path.join(os.path.abspath('E:/graduate_study/work/labelme/a'),
                                       '' + item.split('_')[0] + '.png')#重命名文件
                    os.rename(src, dst)#保存文件


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
# if 'mask' in item:
#     src = os.path.join(os.path.abspath(path), item)
#     dst = os.path.join(os.path.abspath(path), '' + str(int(item[0:3])-90)+'_mask.png')
#     try:
#         os.rename(src, dst)
#         i += 1
#     except:
#         continue
