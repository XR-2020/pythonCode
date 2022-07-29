import os
str='P01-0080.png'
name=str.split('.')[0]
dir_path='E:/python/cnn_demo/picture'
path=os.path.join(dir_path,name)
for i in range(16):
    print('{}/layer1_{}.png'.format(path, 0))
    #
    # for i in range(im.shape[2]):  # 列表中一项里每个特征图生成
    #     # ax = plt.subplot(4, 4, i + 1)
    #     # [H, W, C]  cmap='gray' :设置为灰度图， [:, :, i]选择对channels进行切分
    #     # 设置大小
    #     plt.figure(figsize=(im.shape[0] / 100, im.shape[1] / 100), dpi=100)
    #     plt.axes([0, 0, 1, 1])
    #     plt.imshow(im[:, :, i], cmap='gray')
    #     plt.axis('off')
    #     # 保存图像
    #     if a == 0:
    #         plt.savefig('{}/layer1_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight', pad_inches=0)
    #     if a == 1:
    #         plt.savefig('{}/layer2_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight', pad_inches=0)
    #     if a == 2:
    #         plt.savefig('{}/layer3_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight', pad_inches=0)
    #     if a == 3:
    #         plt.savefig('{}/layer4_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight', pad_inches=0)
    #     if a == 4:
    #         plt.savefig('{}/layer5_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight', pad_inches=0)
    #     if a == 5:
    #         plt.savefig('{}/ravel_layer32_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight',
    #                     pad_inches=0)
    #     if a == 6:
    #         plt.savefig('{}/ravel_layer16_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight',
    #                     pad_inches=0)
    #     if a == 7:
    #         plt.savefig('{}/ravel_layer8_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight',
    #                     pad_inches=0)
    #     if a == 8:
    #         plt.savefig('{}/x32transpose_layer2_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight',
    #                     pad_inches=0)
    #     if a == 9:
    #         plt.savefig('{}/x16 + x32_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight', pad_inches=0)
    #     if a == 10:
    #         plt.savefig('{}/x16transpose_layer2_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight',
    #                     pad_inches=0)
    #     if a == 11:
    #         plt.savefig('{}/x8 + x16_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight', pad_inches=0)
    #     if a == 12:
    #         plt.savefig('{}/transpose_layer8_{}.png'.format(save_path, i + 1), dpi=100, bbox_inches='tight',
    #                     pad_inches=0)
    #     plt.close()