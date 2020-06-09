import torch
from flask import Flask, request, jsonify
import models
import transform

import time
from collections import OrderedDict

from transform import transform_image

app = Flask(__name__)


device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Pytorch facebookresearch-Ws_Images_resnext predict device =',device)

@app.route('/')
def hello():
    return "hello world"


@app.route('/predict', methods=['POST'])
def predict():
    # 获取输入数据
    file = request.files['file']
    # 转换成字节
    img_bytes = file.read()

    # 数据预处理
    image_tensor = transform_image(img_bytes = img_bytes)
    image_tensor = image_tensor.to(device)

    # 模型预测

    # API接口封装

    # Json 格式数据输出
    pass


if __name__ == '__main__':
    app.run()
