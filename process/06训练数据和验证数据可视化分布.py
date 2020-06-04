# -- coding:UTF-8 --
import  os

# 6- 训练数据和验证数据可视化分布

# 目前统计训练和验证数据<label,count>,然后统计图片的方式在一张图中展示

# 获取数据内容
base_path = '../data/'
train_path = os.path.join(base_path,'40_garbage_classify-for-pythorch/train.txt')
val_path = os.path.join(base_path,'40_garbage_classify-for-pythorch/val.txt')

# print('train_path = ',train_path)
# print('val_path = ',val_path)
# train_path =  ../data/40_garbage_classify-for-pythorch/train.txt
# val_path =  ../data/40_garbage_classify-for-pythorch/val.txt

# 统计<label,count>

from glob import glob
import codecs
label_idx_list = []
def get_label_idx_list(data_path):
    label_idx_list = []

    for line in codecs.open(data_path,'r'):
        line = line.strip()
        label_idx = line.split('\t')[1]
        label_idx_list .append(label_idx)
    return label_idx_list

from collections import Counter

# Counter统计label出现次数
# dict类型转换操作
train_dict = dict(Counter(get_label_idx_list(train_path)))

val_dict = dict(Counter(get_label_idx_list(val_path)))

print('train_dict00 = ',train_dict)
print('val_dict00 = ',val_dict)

# 对dict中的key进行sort asc

train_dict = dict(sorted(train_dict.items()))
val_dict = dict(sorted(val_dict.items()))

print('train_dict = ',train_dict)
print('val_dict = ',val_dict)


# 可视化操作
import matplotlib.pyplot as plt

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

plt.figure(figsize=(15, 8))

# 构建数据

# check train == val key
assert train_dict.keys() == val_dict.keys()
x = list(train_dict.keys())

# train
train_y = list(train_dict.values())

# val
val_y = list(val_dict.values())

# 创建Bar示例
width = 0.3
autolabel(plt.bar(x,train_y,width,color='r',label='Train'))
x2 = []
for i in x:
  x2.append(int(i)+width)

autolabel(plt.bar(x2,val_y,width,color='g',label='Val'))
plt.xticks(x, x, rotation=30)  # 这里是调节横坐标的倾斜度，rotation是度数
# 设置全局参数

plt.title('garbage classify Train/Val',color='blue')

# 展示图标
plt.legend()
plt.show()



print("步骤六"+"* "* 60)



