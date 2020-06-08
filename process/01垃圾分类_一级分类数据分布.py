# -- coding:UTF-8 --
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


garbage_classify_index = {"0":"其他垃圾","1":"厨余垃圾","2":"可回收物","3":"有害垃圾"}

garbage_index_classify = {"其他垃圾":"0","厨余垃圾":"1","可回收物":"2","有害垃圾":"3"}

data_list = []
rankl_garbage_classify_rule = {}

for k,v in garbage_classify_rule.items():
    rankl_k = v.split('/')[0]
    rankl_v = k
    # print('rankl_k = ',rankl_k)
    # print('rankl_v = ',rankl_v)
    # print('garbage_index_classify = ',garbage_index_classify[rankl_k])
    data_list.append([rankl_k,int(garbage_index_classify[rankl_k]),int(rankl_v)])

# print('data_list = ',data_list[:2])
# [['其他垃圾', 0, 0], ['其他垃圾', 0, 1]]



# 获取一份分类label 对应的原始数据label

rank_k_v_dict = {}

for data in data_list:
    k = data[2]#原标签
    v = data[1]#新标签
    rank_k_v_dict[k] = v

# print('rank_k_v_dict = ',rank_k_v_dict)
# {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2, 34: 2, 35: 2, 36: 2, 37: 3, 38: 3, 39: 3}

import os

base_path = '../data/'
data_path = os.path.join(base_path,'garbage_classify/train_data')


for (dirpath,dirname,filenames) in os.walk(data_path):
    print("步骤一"+"* "* 60)

# 分析一下*.txt 读取内容，然后img.txt
# 首先匹配一下.txt 文件进行输出


from glob import glob

def get_image_info():
    data_path_txt = os.path.join(data_path,'*.txt')#所有txt文件
    txt_file_list = glob(data_path_txt)
    # print(txt_file_list[:2])

    #存储txt文件变量
    img_path_list = []
    img_name2label_dict = {}
    img_label_dict = {}#<img_label,img_count>

    #读取文件内容
    for file_path in txt_file_list:
        with open(file_path,'r') as f:
            line = f.readline()
            # print ('line = ',line)
            # ['img_4833.jpg, 11']
            # ['img_773.jpg, 1']

            line = line.strip()#前后空格去掉
            img_name = line.split(',')[0]
            img_label = line.split(',')[1]
            img_label = int(img_label)
            # 文件路径../data/garbage_classify/train_data/img_773.jpg
            img_name_path = os.path.join(base_path,'garbage_classify/train_data/{}'.format(img_name))
            # print ('img_name_path = ',img_name_path)
            img_path_list.append({'img_name_path':img_name_path,'img_label':img_label})

            #图片的名称-标签
            img_name2label_dict[img_name] = img_label

            #统计每个标签出现的次数<img_label,img_count>
            img_label_count = img_label_dict.get(img_label,0)
            if img_label_count :
                img_label_dict[img_label] = img_label_count+1
            else:
                img_label_dict[img_label] = 1

    #最终返回的结果
    return img_path_list,img_label_dict,img_name2label_dict


img_path_list,img_label_dict,img_name2label_dict = get_image_info()
print ('img_label_dict = ',img_label_dict)
print ("img_label_dict_len = " ,len(img_label_dict))
print('img_path_list = ',img_path_list)
print('img_name2label_dict = ',img_name2label_dict)

print("步骤二"+"* "* 60)
















