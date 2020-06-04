# -- coding:UTF-8 --

import os


base_path = '../data/'
data_path = os.path.join(base_path,'garbage_classify/train_data')

# for filename in os.listdir(data_path):
#     # print(os.path.join(data_path, filename))
#     print(filename)

for (dirpath,dirname,filenames) in os.walk(data_path):
    print("* "* 60)
    print ('Directory path:',dirpath)
    print('total examples = ',len(filenames))