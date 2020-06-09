import io
import torchvision.transforms as transforms


from PIL import Image

def transform_image(img_bytes):

    '''
    图片数据-》数据预处理

    :param img_bytes:
    :return:
    '''
    preprocess = transforms.Compose([
        # 图像变了：对我们的图片进行缩放
        transforms.Resize(256),
        # 裁剪：中心裁剪
        transforms.CenterCrop(224),
        # 数据归一化操作[0,1]
        transforms.ToTensor(),
        # 图像数据标准化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(img_bytes))

    image_tensor =  preprocess(image)
    # 加个维度
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor