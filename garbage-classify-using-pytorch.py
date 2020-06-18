# -- coding:UTF-8 --

# 1.导入库
#- 系统库
import  os
from os import walk
#-torch相关包
import torch
import torch.nn as nn
from torchvision import datasets
#- 相关参数
from args import args

#- 数据预处理的函数定义部分
from transform import preprocess

#- 模型pre_trained_model 加载、训练、评估、标签映射关系

from model import train,evaluate,initial_model,class_idname


#- 工具栏：日志工具类、模型保存、优化器

from utils.logger import Logger
from utils.misc import save_checkpoint,get_optimizer

#- 训练混淆矩阵效果评估工具栏

from sklearn import metrics




# 2.数据的整体探测（通过目录查询目录下的整体格式）
base_path = 'data/garbage_classify-for-pytorch'
for dirpaths,dirnames,filenamses in walk(base_path):
    if(len(filenamses)>0):
        print('* '*60)
        print('dirpaths = ',dirpaths)
        print('dirnames = ',dirnames)
        print('filenamses = ',filenamses)

# 3.数据封装ImageFolder格式
TRAIN = "{}/train".format(base_path)
VALID = "{}/val".format(base_path)
print('train data_path = ',TRAIN)
print('val data_path = ',VALID)

# root:根目录的路径
train_data = datasets.ImageFolder(root = TRAIN,transform=preprocess)
val_data = datasets.ImageFolder(root = VALID,transform=preprocess)

# print(train_data)
assert train_data.class_to_idx.keys() == val_data.class_to_idx.keys()
print('imgs = ',train_data.imgs[:2])

# 4.批量的数据加载
batch_size = 12
num_workers = 2

from torch.utils.data.dataloader import DataLoader

train_loader = DataLoader(
    train_data,batch_size = batch_size,
    num_workers = num_workers,
    shuffle = True
)

val_loader = DataLoader(
    train_data,batch_size = batch_size,
    num_workers = num_workers,
    shuffle = False
)

# image,label
image,label = next(iter(train_loader))

print(label)
print(image.shape)

# 5.定义模型训练和验证方法

def run(model,train_loader,val_loader):
    '''
    模型训练和预测
    :param model: 初始化的model
    :param train_loader: 训练集数据
    :param val_loader: 验证集数据
    :return:
    '''

    # 初始化变量参数

    # 模型保存的变量
    global best_acc
    # 训练C类别的分类问题，我们使用CrossEntropyloss
    criterion = nn.CrossEntropyLoss()
    # torch.optim 是一个各种优化算法库
    # optimizer对象能保存当前的参数状态并且基于计算梯度更新参数
    optimizer = get_optimizer(model,args)



    #加载checkpoint信息：model的断点续传 ，可以指定迭代的开始位置进行重新训练



    # 模型评估：混淆矩阵；准确率、召回率、F1-score


    # 模型的训练和验证
    for epoch in range(1,args.epochs+1):
        # train
        train(train_loader,model,criterion,optimizer)

        # val

        pass

    pass


# 入口程序

if __name__=='__main__':
    print('hello')

    # 模型初始化
    model_name = args.model_name
    num_classes = args.num_classes
    # 初始化模型
    model_ft = initial_model(model_name,num_classes,feature_extract = True)

    print(model_ft)

    #模型训练和评估
    run(model_ft,train_loader,val_loader)


















