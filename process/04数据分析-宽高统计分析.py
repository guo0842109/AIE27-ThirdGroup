# -- coding:UTF-8 --

import os

# 4-数据分析 - 高度与宽度的统计分析
# 数据长度和挂镀分布比例

# python如何获取JPG图片的长度和高度信息
# 我们通过PIL中Image类处理

# 导入PIL库
from PIL import Image


# 获取一张图片
base_path = '../data/'

data_path = os.path.join(base_path,'garbage_classify/train_data')
pic_file_name = os.path.join(data_path,'img_1.jpg')

print('pic_file_name = ',pic_file_name)

# 获取长度和高度
img = Image.open(pic_file_name)
# img.size =  (800, 575)
# print('img.size = ', img.size)


# 那么，接下来，我们来统计分析一些所有JPG图片，来获取统计信息

from glob import glob


data_path_jpg = os.path.join(data_path,'*.jpg')

# print(data_path_jpg)
# ../data/garbage_classify/train_data/*.jpg

# 扫描所有jpg图片
data_path_list = glob(data_path_jpg)
# print(data_path_list)


for file_path in data_path_list[:2]:
    print('file_path = ',file_path)
    # file_path =../ data / garbage_classify / train_data / img_10486.jpg
    img = Image.open(file_path)

    width = img.size[0]
    height = img.size[1]

    ratio = float("{:.2f}".format(width/height))

    img_name = file_path.split('/')[-1]
    img_id = img_name.split('.')[0].split('_')[-1]
    img_label = img_name2label_dict.get(img_name,0)
    print('width,height,ratio = ',width,height,ratio)
    # width, height, ratio = 443 264 1.68


