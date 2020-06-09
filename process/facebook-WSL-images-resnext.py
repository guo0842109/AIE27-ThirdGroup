# -- coding:UTF-8 --

# 1-导入库
# 导入库
import torch
import torch.nn as nn
from torchvision import transforms

# 导入我们自定义的model
import models

# 2-加载模型

model_ft = models.resnext101_32x16d_wsl()
# 3-预测问题，置顶eval
r = model_ft.eval()
# 接下来看下，我们看下我们的模型参数
print(model_ft.parameters())

# 4- 加载图片数据
file_name = 'images/yindu.jpg'
from PIL import Image
input_image= Image.open(file_name)
print(input_image)
print(input_image.size)


# 数据可视化
import matplotlib.pyplot as plt

plt.imshow(input_image)
plt.show()

# 5 - 图片数据预处理

preprocess = transforms.Compose([
    # 图像变了：对我们的图片进行缩放
    transforms.Resize(256),
    # 裁剪：中心裁剪
    transforms.CenterCrop(224),
    # 数据归一化操作[0,1]
    transforms.ToTensor(),
    #图像数据标准化
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

input_tensor = preprocess(input_image)

print('input_tensor.shape = ',input_tensor.shape)
print('input_tensor = ',input_tensor)


# 转换格式

input_batch = input_tensor.unsqueeze(0)
print('convert data format input_batch = ',input_batch.shape)

# 数据预处理后，展示
import matplotlib.pyplot as plt
image_tmp = input_tensor.permute(1,2,0)
input_tensor = torch.clamp(image_tmp,0,1)
plt.imshow(image_tmp)
plt.show()

# 6- 模型在线预测
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model_ft.to('cuda')

with torch.no_grad():
    output = model_ft(input_batch)


print(output[0])
print(output[0].shape)
print(torch.nn.Softmax(output[0],dim=0))


# 7-获取最大可能性的类别
# 需要获取的数据：id-name的标签，获取结果中最大可能性的id号

result = torch.nn.Softmax(output[0],dim=0)

# result -> list
v_list = result.cpu().numpy().tolist()

v_max = 0
idx = 0

for i,v in enumerate(v_list):
    if v > v_max:
        v_max= v
        idx = i
print('v_max = ',v_max)#1000个分类中,idx对应的可能性取值
print('idx = ',idx)

# ...

import codecs
ImageNet_dict = {}

for line in  codecs.open('../data/...','r'):
    line = line.strip()
    print(line)
    _id = line.split(":")[0]
    _name = line.split(":")[1]
    _name = _name.replace("\xa0","")
    ImageNet_dict[int[_id]] = _name


print(ImageNet_dict[idx])






