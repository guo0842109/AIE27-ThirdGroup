# 二级分类数据转一级分类数据
garbage_classify_rule = {
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

garbage_classify_index = {"0": "其他垃圾", "1": "厨余垃圾", "2": "可回收物", "3": "有害垃圾"}
garbage_index_classify = {"其他垃圾": "0", "厨余垃圾": "1", "可回收物": "2", "有害垃圾": "3"}

data_list = []
rank1_garbage_classify_rule = {}
for k, v in garbage_classify_rule.items():
    rank1_k = v.split('/')[0]
    rank1_v = k
    data_list.append([rank1_k, int(garbage_index_classify[rank1_k]), int(rank1_v)])

# 获取一级分类label 对应的原始数据label
rank_k_v_dict = {}
for data in data_list:
    k = data[2]  # 原标签
    v = data[1]  # 新标签
    rank_k_v_dict[k] = v
print(rank_k_v_dict)

# k_vlist_dict 就是一级分类需要处理的结果


# 整体数据探测

import os
from os import walk

base_path = '../data/'
data_path = os.path.join(base_path,'garbage_classify/train_data')
for (dirpath, dirnames, filenames) in walk(data_path):
    if len(filenames)>0:
        print('*'*60)
        print("Directory path: ", dirpath)
        print("total examples: ", len(filenames))
        print("File name Example: ", filenames[:5])

        # 我们来分析 *.txt读取内容，然后获取img.txt
        #
        # 首先，我们需要
        # 匹配txt
        # 文件进行输出

from glob import glob
import os


def get_img_info():
    data_path_txt = os.path.join(data_path, '*.txt')
    txt_file_list = glob(data_path_txt)

    # 存储txt 文件
    img_path_txt = 'img.txt'
    img_path_list = []
    img_label_dict = dict()  # <标签，次数>
    img_name2label_dict = {}
    for file_path in txt_file_list:
        with open(file_path, 'r') as f:
            line = f.readline()

        line = line.strip()
        img_name = line.split(',')[0]
        img_label = line.split(',')[1]
        img_label = int(img_label.strip())
        # 图片路径＋标签
        img_name_path = os.path.join(base_path, 'garbage_classify/train_data/{}'.format(img_name))
        img_path_list.append(
            {'img_name_path': img_name_path,
             'img_label': img_label})
    return img_path_list

print('img_path_list = ',get_img_info()[:10])

# 对img_path_list 的img_label 进行修改为一级分类的标签
img_path_list = []
img_label_dict = {}
for img_info in get_img_info():
    img_label = img_info['img_label']  # 修正前的标签
    img_label = rank_k_v_dict[img_label]
    img_info.update({'img_label': img_label})  # 修正后的标签

    # 图片路径＋标签
    img_path_list.append(img_info)

    # 统计每个标签出现次数
    img_label = int(img_label)
    img_label_count = img_label_dict.get(img_label, 0)
    if img_label_count:
        img_label_dict[img_label] = img_label_count + 1
    else:
        img_label_dict[img_label] = 1

print('img_path_list = ', img_path_list[:3])
print('img_label_dict = ', img_label_dict)

# 数据不同类别分布
# 我们这里通过柱状图 来分析不同类别分布情况，我们使用pyecharts 这种效果非常棒的工具来展示
#
# x 轴数据： 标签名称
#
# y 轴数据： 标签名称对应次数
#
# 首先我们 img_label_dict 按照key 进行排序，这样方便查看数据

img_label_dict = dict(sorted(img_label_dict.items()))
print(img_label_dict)
print(garbage_classify_index)
print([garbage_classify_index[str(k)] for k in img_label_dict.keys()])
print(list(img_label_dict.values()))

# 导入库
from pyecharts import  options as opts
from pyecharts.charts import Bar

# 构建满足pyecharts 格式数据
x = [garbage_classify_index[str(k)] for k in img_label_dict.keys()]

y = list(img_label_dict.values())

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
# plt.xticks(x, x, rotation=30)  # 这里是调节横坐标的倾斜度，rotation是度数
autolabel(plt.bar(x,y,color='rgb', tick_label=x))
plt.show()


# {'0': '其他垃圾', '1': '厨余垃圾', '2': '可回收物', '3': '有害垃圾'}
# ['其他垃圾', '厨余垃圾', '可回收物', '有害垃圾']
# [1652, 3389, 8611, 1150]

# 数据分析，可以得出一下的结论：
#
# 公共4 个分类，如上图分析Bar 图所示
#
# 较少数据为其他垃圾
#
# 较多的数据类别可以回收的垃圾
#
# 我们的模型通过深度学习的迁移模型来完成，小数据量的样本也可以达到很好的效果，这些数据可以直接参与模型的训练输入

# 切分训练集和测试集

len(img_path_list)
img_path_list[0]
import random
random.shuffle(img_path_list)

# 0.8 0.2 切分
img_count = len(img_path_list)
train_img_list = img_path_list[:int(img_count*0.8)]
val_img_list = img_path_list[int(img_count*0.8):]

print('train_size=',len(train_img_list))
print('valid_size=',len(val_img_list))
# 数据切分后，我们生成训练和验证集的数据
import shutil

# 训练数据处理
with open(os.path.join(base_path, '4_garbage-classify-for-pytorch/train.txt'), 'w') as f:
    for img_dict in train_img_list:
        # 文本格式数据
        img_name_path = img_dict['img_name_path']  # ../data/garbage_classify/img_11674.jpg
        img_label = img_dict['img_label']
        f.write("{}\t{}\n".format(img_name_path, img_label))
        # 图片-标签目录
        garbage_classify_dir = os.path.join(base_path, '4_garbage-classify-for-pytorch/train/{}'.format(img_label))
        # print(garbage_classify_dir)
        if not os.path.exists(garbage_classify_dir):
            os.makedirs(garbage_classify_dir)
        # 拷贝数据到目录下
        # print(garbage_classify_dir,img_name_path)
        shutil.copy(img_name_path, garbage_classify_dir)

# 验证数据处理
with open(os.path.join(base_path, '4_garbage-classify-for-pytorch/val.txt'), 'w') as f:
    for img_dict in val_img_list:
        # 文本格式数据
        img_name_path = img_dict['img_name_path']  # ../data/garbage_classify/img_11674.jpg
        img_label = img_dict['img_label']
        f.write("{}\t{}\n".format(img_name_path, img_label))
        # 图片-标签目录
        garbage_classify_dir = os.path.join(base_path, '4_garbage-classify-for-pytorch/val/{}'.format(img_label))
        # print(garbage_classify_dir)
        if not os.path.exists(garbage_classify_dir):
            os.makedirs(garbage_classify_dir)
        # 拷贝数据到目录下
        # print(garbage_classify_dir,img_name_path)
        shutil.copy(img_name_path, garbage_classify_dir)

# 最后，我们来分析下。切分后验证集和训练集的分布

train_path = os.path.join(base_path,'4_garbage-classify-for-pytorch/train.txt')
test_path = os.path.join(base_path,'4_garbage-classify-for-pytorch/val.txt')
print(train_path)
print(test_path)


def get_label_idx_list(data_path):
    label_idx_list = []
    import codecs
    for line in codecs.open(data_path,'r'):
        line = line.strip()
        label_idx = line.split('\t')[1]
        label_idx_list.append(int(label_idx))
    return label_idx_list


from collections import Counter
train_dict = dict(Counter(get_label_idx_list(train_path)))
test_dict = dict(Counter(get_label_idx_list(test_path)))

train_dict = dict(sorted(train_dict.items()))
test_dict = dict(sorted(test_dict.items()))

print("train_dict = ",train_dict)
print("test_dict = ",test_dict)
print('garbage_classify_index = ',garbage_classify_index)

# 导入库

import matplotlib.pyplot as plt

assert len(train_dict.keys())==len(test_dict.keys())
x = [ "{}-{}".format(label_idx, garbage_classify_index.get(str(label_idx),""))
     for label_idx in img_label_dict.keys()]
x = list(x)
y = list(train_dict.values())

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

plt.figure(figsize=(15, 8))

plt.title('garbage classify',color='blue')
# plt.xticks(x, x, rotation=30)  # 这里是调节横坐标的倾斜度，rotation是度数
autolabel(plt.bar(x,y,color='rgb', tick_label=x))
plt.show()
