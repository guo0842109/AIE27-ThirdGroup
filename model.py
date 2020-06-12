import models
import torch
import torch.nn as nn


def train():
    pass


def evaluate():
    pass


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


def initial_model(model_name, num_classes, feature_extract=True):
    '''
    基于pre_trained_model进行初始化
    :param model_name:
    提供的模型名称：如：resnext101_32x16d/resnext101_32x8d
    :param num_classes: 图片分类个数
    :param feature_extract: 设置true，固定特征提取层，优化全连接的分类器
    :return:
    '''
    model_ft = None
    if (model_name == 'resnext101_32x16d'):
        # 加载Facebook pre_trained_model restnext101，默认1000类
        model_ft = models.resnext101_32x16d_wsl()
        # 设置 固定特征提取层
        set_parameter_requries_grad(model_ft, feature_extract)
        # 调整分类的个数
        num_ftrs = model_ft.fc.in_features
        # 修复fc的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=num_ftrs, out_features=num_classes)
        )
    elif (model_name == 'resnext101_32x8d'):
        # 加载Facebook pre_trained_model restnext101，默认1000类
        model_ft = models.resnext101_32x8d_wsl()
        # 设置 固定特征提取层
        set_parameter_requries_grad(model_ft, feature_extract)
        # 调整分类的个数
        num_ftrs = model_ft.fc.in_features
        # 修复fc的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=num_ftrs, out_features=num_classes)
        )
    else:
        print('Invalid mode name,exting...')
        exit()

    return model_ft


def class_idname():
    pass
