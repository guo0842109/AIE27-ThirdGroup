# -- coding:UTF-8 --

# 过程包括：缩放、裁剪、归一化、标准化


# 导入库
# pytorch相关库
import matplotlib

import torch
import torchvision
from torchvision import transforms

print('torch.__version__ = ',torch.__version__)
print('torchvision.__version__ = ',torchvision.__version__)

# 3-加载图片数据

file_name= "../data/40_garbage_classify-for-pythorch/tarin/0/img_4.jpg"

from PIL import Image
input_image = Image.open(file_name)

# 图片的尺寸
print('input_image.shape = ',input_image.size)
# input_image.shape =  (1024, 942)

# 数据处理后，我们通过可视化来进行效果展示

import matplotlib.pyplot as plt

plt.imshow(input_image)
plt.show()

# 4-pytorch 数据预处理
# 定义pytorch预处理方法

preprocess = transforms.Compose([
    # 图像变了：对我们的图片进行缩放
    transforms.Resize((256,256)),
    # 裁剪：中心裁剪
    transforms.CenterCrop(224),
    # 数据归一化操作[0,1]
    transforms.ToTensor(),
    #图像数据标准化
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

])

# 5-原始数据-预处理操作

input_tensor = preprocess(input_image)

print('input_tensor.shape = ',input_tensor.shape)
# input_tensor.shape =  torch.Size([3, 256, 256])
print(input_tensor)

#6- 数据的可视化展示

# 长和宽交换，目的是展示
input_tensor = input_tensor.permute(1,2,0)
print('input_tensor.shape = ',input_tensor.shape)
# input_tensor.shape =  torch.Size([256, 256, 3])
plt.imshow(input_tensor)
plt.show()
















