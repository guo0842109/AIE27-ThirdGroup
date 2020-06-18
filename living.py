# -- coding:UTF-8 --

# 如何根据pre_trained_model 跳转业务需要的model

import torch
import torch.nn as nn

import models

def set_parameter_requries_grad(model, feature_extract):
    '''

    :param model: 模型
    :param feature_extract: true，固定特征抽取层
    :return:
    '''
    if feature_extract:
        for parm in model.parameters():
            # 不需要更新梯度，冻结某些层的梯度
            parm.requires_grad = False


model_ft = models.resnext101_32x16d_wsl()

# model的网络结构
# print(model_ft)

# 我们的目的是修改输出的类别
print(model_ft.fc)

model_ft.fc.in_features,model_ft.fc.out_features

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Sequential(
    nn.Linear(in_features=num_ftrs,out_features=4)
)



