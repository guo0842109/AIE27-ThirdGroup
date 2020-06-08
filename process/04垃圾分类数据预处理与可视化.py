# -- coding:UTF-8 --

# 数据探测

import numpy as no
import os
import time

print(os.listdir('../data/40_garbage_classify-for-pythorch'))

from os import walk

# for(dirpath,dirname,filenames) in walk('../data/40_garbage_classify-for-pythorch'):
#     print("* " * 60)
#     print("Director path = ",dirpath)
#     print("total examples = ",len(filenames))
#     print("File name Examples = ",filenames[:5])


#2数据预处理

label_dict = {
    "0": "其他垃圾/一次性快餐盒",
    "1": "其他垃圾/污损塑料",
    "2": "其他垃圾/烟蒂",
    "3": "其他垃圾/牙签",
    "4": "其他垃圾/破碎花盆及碟碗",
    "5": "其他垃圾/竹筷",
    "6": "厨余垃圾/剩饭剩菜",
    "7": "厨余垃圾/大骨头",
    "8": "厨余垃圾/水果果皮",
    "9": "厨余垃圾/水果果肉",
    "10": "厨余垃圾/茶叶渣",
    "11": "厨余垃圾/菜叶菜根",
    "12": "厨余垃圾/蛋壳",
    "13": "厨余垃圾/鱼骨",
    "14": "可回收物/充电宝",
    "15": "可回收物/包",
    "16": "可回收物/化妆品瓶",
    "17": "可回收物/塑料玩具",
    "18": "可回收物/塑料碗盆",
    "19": "可回收物/塑料衣架",
    "20": "可回收物/快递纸袋",
    "21": "可回收物/插头电线",
    "22": "可回收物/旧衣服",
    "23": "可回收物/易拉罐",
    "24": "可回收物/枕头",
    "25": "可回收物/毛绒玩具",
    "26": "可回收物/洗发水瓶",
    "27": "可回收物/玻璃杯",
    "28": "可回收物/皮鞋",
    "29": "可回收物/砧板",
    "30": "可回收物/纸板箱",
    "31": "可回收物/调料瓶",
    "32": "可回收物/酒瓶",
    "33": "可回收物/金属食品罐",
    "34": "可回收物/锅",
    "35": "可回收物/食用油桶",
    "36": "可回收物/饮料瓶",
    "37": "有害垃圾/干电池",
    "38": "有害垃圾/软膏",
    "39": "有害垃圾/过期药物"
}
# 2.1导入图像分类库
import torch
from torchvision import datasets,models,transforms
from matplotlib import  pyplot as plt
# 解决中文乱码的问题
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 2.2定义数据输入

# 数据tensor运行方式：cpu or  cuda
device = torch.device('coda' if torch.cuda.is_available() else 'cpu')

print('device = ',device)

# 定义数据路径
TRAIN = '../data/40_garbage_classify-for-pythorch/tarin'
VAL = '../data/40_garbage_classify-for-pythorch/val'

print('train_path = ',TRAIN)
print('val_path = ',VAL)

# device =  cpu
# train_path =  ../data/40_garbage_classify-for-pythorch/train
# val_path =  ../data/40_garbage_classify-for-pythorch/val

# 2.3原始垃圾数据展示
train_data = datasets.ImageFolder(TRAIN)
val_data = datasets.ImageFolder(VAL)


from PIL import Image
fig = plt.figure(figsize=(25,8))
print('train_data.imgs = ',train_data.imgs[:4])

# train_data.imgs中的每张图片，进行可视化展示，看看原始数据的样子
for idx,img in enumerate(train_data.imgs[:18]):
    # ax 图表示例
    ax = fig.add_subplot(3,18/3,idx+1,xticks = [],yticks = [])

    file_name_path = img[0]
    target_idx = img[1]
    print(idx,img,file_name_path,target_idx)
    target_name = label_dict.get(str(target_idx),"")

#     绘制图表
    image = Image.open(file_name_path)

    ax.set_title('{}-{}'.format(target_idx,target_name))
    plt.imshow(image)
    plt.show()












