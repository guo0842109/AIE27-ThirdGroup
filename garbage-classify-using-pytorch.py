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




# 2.数据的整体探测
base_path = ''
# 3.数据封装ImageFolder格式

# 4.批量的数据加载

# 5.定义模型训练和验证方法


# 入口程序

if __name__=='__main__':
    print('hello')