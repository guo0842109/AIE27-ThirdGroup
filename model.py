# 导入模型定义方法
import models
# 导入Torch先关包
import torch
import torch.nn as nn

# 导入工具类
from utils.eval import accuracy
from utils.misc import AverageMeter

# 导入进度条
from progress.bar import Bar

import time

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_loader, model, criterion, optimizer):
    '''
    模型训练
    :param train_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :return:
    '''

    # 定义保存更新变量
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    # # # # # # #
    # train the model
    # # # # # # #
    model.train()
    # 循环每批数据，然后进行模型的训练

    # 定义bar变量
    bar = Bar('processing',max =len(train_loader) )

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensors to GPU if cuda is_available
        inputs, targets = inputs.to(device), targets.to(device)
        # 在进行反向传播之前，我们使用zero_grad方法进行清空梯度
        optimizer.zero_grad()
        # 模型的预测
        outputs = model(inputs)
        # 计算loss
        loss = criterion(outputs, targets)

        # 计算acc和变量的更新操作
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 1))

        # backward pass
        loss.backward()
        # perform as single optimization step(paramter update)
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        # 把主要的参数打包放进bar
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
            batch=batch_index + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg
        )
        bar.next()
    return (losses.avg, top1.avg)


def evaluate(val_loader,model,criterion,test = None):
    '''
    模型评估
    :param val_loader:
    :param model:
    :param criterion:
    :param test:
    :return:
    '''

    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    predict_all  = np.array([],dtype=int)
    labels_all = np.array([],dtype=int)
    # # # # # # #
    # val the model
    # # # # # # #
    model.eval()

    # 循环每批数据，然后进行模型的训练
    # 定义bar变量
    bar = Bar('processing', max=len(val_loader))
    for batch_index, (inputs, targets) in enumerate(val_loader):
        data_time.update(time.time() - end)
        # move tensors to GPU if cuda is_available
        inputs, targets = inputs.to(device), targets.to(device)

        # 模型的预测
        outputs = model(inputs)
        # 计算loss
        loss = criterion(outputs, targets)

        # 计算acc和变量的更新操作
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 1))


        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        # 评估混淆矩阵的数据
        targets = targets.data.cpu().numpy()#真实数据的y数值
        predict = torch.max(outputs.data,1)[1].cpu().numpy()#预测数据y数值
        labels_all = np.append(labels_all,targets)#数据赋值
        predict_all = np.append(predict_all,predict)

        # plot progress
        # 把主要的参数打包放进bar
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
            batch=batch_index + 1,
            size=len(val_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg
        )
        bar.next()


    if test:
        return (losses.avg, top1.avg,predict_all,labels_all)
    return (losses.avg, top1.avg)

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
