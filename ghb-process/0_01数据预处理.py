# -- coding:UTF-8 --

import os


garbage_classify_rule= {
    "0": "其他垃圾/一次性快餐盒",
    "1": "其他垃圾/污损塑料",
    "2": "其他垃圾/烟蒂",
    "3": "其他垃圾/牙签",
    "4": "其他垃圾/破碎花盆及碟碗",
    "5": "其他垃圾/竹筷",
    "6": "厨余垃圾/剩饭剩菜",
    "7": "厨余垃圾/大骨头",
    "8": "厨余垃圾/水果果皮",
    "9": "厨余垃圾/水果果肉",
    "10": "厨余垃圾/茶叶渣",
    "11": "厨余垃圾/菜叶菜根",
    "12": "厨余垃圾/蛋壳",
    "13": "厨余垃圾/鱼骨",
    "14": "可回收物/充电宝",
    "15": "可回收物/包",
    "16": "可回收物/化妆品瓶",
    "17": "可回收物/塑料玩具",
    "18": "可回收物/塑料碗盆",
    "19": "可回收物/塑料衣架",
    "20": "可回收物/快递纸袋",
    "21": "可回收物/插头电线",
    "22": "可回收物/旧衣服",
    "23": "可回收物/易拉罐",
    "24": "可回收物/枕头",
    "25": "可回收物/毛绒玩具",
    "26": "可回收物/洗发水瓶",
    "27": "可回收物/玻璃杯",
    "28": "可回收物/皮鞋",
    "29": "可回收物/砧板",
    "30": "可回收物/纸板箱",
    "31": "可回收物/调料瓶",
    "32": "可回收物/酒瓶",
    "33": "可回收物/金属食品罐",
    "34": "可回收物/锅",
    "35": "可回收物/食用油桶",
    "36": "可回收物/饮料瓶",
    "37": "有害垃圾/干电池",
    "38": "有害垃圾/软膏",
    "39": "有害垃圾/过期药物"
}

garbage_classify_dict = {}
for k,v in garbage_classify_rule.items():
    rankl_k = v.split('/')[0]
    rankl_v = k
    garbage_classify_dict[rankl_k] = rankl_v
print(garbage_classify_dict)

base_path = '../data/'
data_path = os.path.join(base_path,'garbage_classify/train_data')

# print(data_path)

for (dirpath,dirname,filenames) in os.walk(data_path):
    print()
    # print('dirpath = ',dirpath)
    # print('dirname = ',dirname)
    # print('filenames = ',filenames)


from glob import glob

img_path_list = []

img_name2label_dict = {}

img_label_dict = {}


def get_image_info():
    # 获取所有txt的路径
    data_path_txt = os.path.join(data_path,'*.txt')
    # print('data_path_txt = ',data_path_txt)
    # 生成txt的文件路径的list
    data_list_txt = glob(data_path_txt)
    # print('data_list_txt = ', data_list_txt)
    # 读取txt文件
    for file_path in data_list_txt:
        with open(file_path,'r') as f:
            line = f.readline()
            # print('line = ',line)
            line = line.strip()
            img_name = line.split(',')[0]
            img_label = line.split(',')[1]
            img_label = int(img_label)
            img_name_path = os.path.join(data_path,img_name)
            # print('img_name = ',img_name)
            # print('img_label = ',img_label)
            # print('img_name_path = ',img_name_path)
            img_path_list.append({"img_name_path":img_name_path,"img_label":img_label})
            img_name2label_dict[img_name] = img_label

            img_label_count = img_label_dict.get(img_label)
            if img_label_count :
                img_label_dict[img_label] = img_label_count +1
            else:
                img_label_dict[img_label] = 1

    return img_path_list,img_name2label_dict,img_label_dict


img_path_list,img_name2label_dict,img_label_dict = get_image_info()

# print('img_path_list = ',img_path_list)
# print('img_name2label_dict = ',img_name2label_dict)
# print('img_label_dict = ',img_label_dict)

x = []
for k in img_label_dict.keys():
    x.append("{}-{}".format(k,garbage_classify_dict.get(k)))


x = list(x)
y = list(img_label_dict.values())


import matplotlib.pyplot as plt
# 显示高度
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

plt.figure(figsize=(15, 8))

plt.title('garbage classify_me',color='blue')
plt.xticks(x, y, rotation=30)  # 这里是调节横坐标的倾斜度，rotation是度数
autolabel(plt.bar(x,y,color='rgb', tick_label=x))
plt.show()