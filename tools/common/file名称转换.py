import os


#将某个文件夹下的文件重命名
class BatchRename():

    def rename(self):
          path = r"E:/python/MyUnet/result/"
          filelist = os.listdir(path)
          total_num = len(filelist)
          i = 1
          for item in filelist:
              print(item)
              # if 'mask' in item:
              #     src = os.path.join(os.path.abspath(path), item)
              #     dst = os.path.join(os.path.abspath(path), '' + str(int(item[0:3])-90)+'_mask.png')
              #     try:
              #         os.rename(src, dst)
              #         i += 1
              #     except:
              #         continue
              if item.endswith('.png'):
                  src = os.path.join(os.path.abspath(path), item)
                  dst = os.path.join(os.path.abspath('E:/python/MyUnet/result/'), '' + item.split('_')[0] + '.png')
                  try:
                      os.rename(src, dst)
                      i += 1
                  except:
                    continue


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
