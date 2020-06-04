# -- coding:UTF-8 --

import os

base_path = '../data/'
data_path = os.path.join(base_path,'garbage_classify/train_data')

# 数据路径
# print ('data_path = ',data_path)
# ('data_path = ', '../data/garbage_classify/train_data')

for (dirpath,dirname,filenames) in os.walk(data_path):
    print("步骤一"+"* "* 60)
    # print ('Directory path:',dirpath)
    # print('total examples = ',len(filenames))
    # print ('file name example:',filenames[:5])#文件列表

    # ('Directory path:', '../data/garbage_classify/train_data')
    # ('total examples = ', 29604)
    # ('file name example:', ['img_4833.txt', 'img_11798.jpg', 'img_10486.jpg', 'img_773.txt', 'img_3184.txt'])

# 分析一下*.txt 读取内容，然后img.txt
# 首先匹配一下.txt 文件进行输出


from glob import glob

def get_image_info():
    data_path_txt = os.path.join(data_path,'*.txt')#所有txt文件
    txt_file_list = glob(data_path_txt)
    # print(txt_file_list[:2])

    #存储txt文件变量
    img_path_list = []
    img_name2label_dict = {}
    img_label_dict = {}#<img_label,img_count>

    #读取文件内容
    for file_path in txt_file_list:
        with open(file_path,'r') as f:
            line = f.readline()
            # print ('line = ',line)
            # ['img_4833.jpg, 11']
            # ['img_773.jpg, 1']

            line = line.strip()#前后空格去掉
            img_name = line.split(',')[0]
            img_label = line.split(',')[1]
            img_label = int(img_label)
            # 文件路径../data/garbage_classify/train_data/img_773.jpg
            img_name_path = os.path.join(base_path,'garbage_classify/train_data/{}'.format(img_name))
            # print ('img_name_path = ',img_name_path)
            img_path_list.append({'img_name_path':img_name_path,'img_label':img_label})

            #图片的名称-标签
            img_name2label_dict[img_name] = img_label

            #统计每个标签出现的次数<img_label,img_count>
            img_label_count = img_label_dict.get(img_label,0)
            if img_label_count :
                img_label_dict[img_label] = img_label_count+1
            else:
                img_label_dict[img_label] = 1

    #最终返回的结果
    return img_path_list,img_label_dict,img_name2label_dict


img_path_list,img_label_dict,img_name2label_dict = get_image_info()
# print (img_label_dict)
print ("img_label_dict_len = " ,len(img_label_dict))
# print(img_path_list)
# print(img_name2label_dict)

print("步骤二"+"* "* 60)

# 3-数据不同类别分布
# 首先构建数据

x = img_label_dict.keys()
# print ('x = ',x)
y = img_label_dict.values()
# print ('y = ',y)

# 通过 matplotlib 来构建图标
#导入matplotlib相关库

# 构建pyecharts需要的数据
x = list(x)
y = list(y)

import matplotlib.pyplot as plt

# 显示高度
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

plt.figure(figsize=(15, 8))

plt.title('garbage classify',color='blue')
plt.xticks(x, x, rotation=30)  # 这里是调节横坐标的倾斜度，rotation是度数
autolabel(plt.bar(x,y,color='rgb', tick_label=x))
plt.show()

# 较小数据是3牙签、20快递纸袋
# 较多数据是11菜叶子、21插头电线
# 数据不均衡，最少跟最大相差10倍，数据不均衡对数据有一定影响
# 最好的方法，扩充数据

print("步骤三"+"* "* 60)

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

# print('pic_file_name = ',pic_file_name)

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
data_list = []
# print(data_path_list)


for file_path in data_path_list:
    # print('file_path = ',file_path)
    # file_path =../ data / garbage_classify / train_data / img_10486.jpg
    img = Image.open(file_path)

    width = img.size[0]
    height = img.size[1]

    ratio = float("{:.2f}".format(width/height))

    img_name = file_path.split('/')[-1]
    img_id = img_name.split('.')[0].split('_')[-1]
    img_label = img_name2label_dict.get(img_name,0)
    # print('width,height,ratio = ',width,height,ratio)
    # width, height, ratio = 443 264 1.68
    # print(int(img_id),width,height,ratio,int(img_label))
    data_list.append([int(img_id),width,height,ratio,int(img_label)])

# print(data_list[:3])


# 首先分析单变量进行数据分析，这个时候我们使用直方图来完成
# python中的seaborn可视化工具进行展示

import numpy as np
import seaborn as sns

ratio_list = [ratio[3] for ratio in data_list]

print(ratio_list[:4])
# 14192

sns.set()
np.random.seed(0)

# seaborn直方图展示

sns.distplot(ratio_list)
plt.show()
#结果显示数据分布在(0,2)

# 接下来获取(0,2)数据，查看效果

# 需要过滤ratio_list的数据，list类型

ratio_list = list(filter(lambda x:x>0.5 and x<=2,ratio_list))
print(ratio_list)
print(len(ratio_list))

# 过滤后的数据，再次显示
import numpy as np
import seaborn as sns

# seaborn直方图展示

sns.distplot(ratio_list)
plt.show()

# 从上面内容可知，数据整天分布在(0.5,1.5)之间
print("步骤四"+"* "* 60)

#5- 切分数据集 - 训练集和验证集



print("步骤五"+"* "* 60)

# 原始数据格式
# print(img_path_list[:2])
# [{'img_name_path': '../data/garbage_classify/train_data/img_4833.jpg', 'img_label': 11}, {'img_name_path': '../data/garbage_classify/train_data/img_773.jpg', 'img_label': 1}]


# 原始数据大小
# print(len(img_path_list))
# 14802

import random
# 对原始数据进行随机打散
random.shuffle(img_path_list)

# 数据分布 0.8 ~ 0.2
img_count = len(img_path_list)
train_size = int(img_count *0.8)
print('img_count = ',img_count)
print('train_size = ',train_size)

train_img_list = img_path_list[:train_size]
val_img_list = img_path_list[train_size:]

print('train_img_list size = ',len(train_img_list))
print('val_img_list size = ',len(val_img_list))
# img_count =  14802
# train_size =  11841
# train_img_list size =  11841
# val_img_list size =  2961

# 数据切人后，我们就可以生成训练和验证数据了

# import shutil
# # 训练数据的生成
# with open(os.path.join(base_path,'40_garbage_classify-for-pythorch/train.txt'),'w') as f:
#     for img_dict in train_img_list:
#         # 文本格式数据
#         img_name_path = img_dict['img_name_path']
#         img_label = img_dict['img_label']
#         f.write("{}\t{}\n".format(img_name_path,img_label))
#
#         # 图片标签目录
#         garbage_classifiy_dir = os.path.join(base_path,'40_garbage_classify-for-pythorch/tarin/{}'.format(img_label))
#
#         # 目录创建
#         if not os.path.exists(garbage_classifiy_dir):
#             os.makedirs(garbage_classifiy_dir)
#
#         # 图片数据进行copy
#         shutil.copy(img_name_path,garbage_classifiy_dir)
#
#
#
# # 验证数据的生成
# with open(os.path.join(base_path,'40_garbage_classify-for-pythorch/val.txt'),'w') as f:
#     for img_dict in val_img_list:
#         # 文本格式数据
#         img_name_path = img_dict['img_name_path']
#         img_label = img_dict['img_label']
#         f.write("{}\t{}\n".format(img_name_path,img_label))
#
#         # 图片标签目录
#         garbage_classifiy_dir = os.path.join(base_path, '40_garbage_classify-for-pythorch/val/{}'.format(img_label))
#
#         # 目录创建
#         if not os.path.exists(garbage_classifiy_dir):
#             os.makedirs(garbage_classifiy_dir)
#
#         # 图片数据进行copy
#         shutil.copy(img_name_path, garbage_classifiy_dir)


print("步骤六"+"* "* 60)

# 6- 训练数据和验证数据可视化分布

# 目前统计训练和验证数据<label,count>,然后统计图片的方式在一张图中展示



















