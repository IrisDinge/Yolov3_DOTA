import pandas as pd
from sklearn.utils import shuffle

data = pd.read_csv("/home/dingjin/Yolov3_DOTA/yolo/dota800/DOTA/ImageSets/Main/all.txt")
print(data)
data = shuffle(data)

train = data.iloc[0:18019]
val = data.iloc[18019:24019]
test = data.iloc[24019:30019]

train.to_csv("/home/dingjin/Yolov3_DOTA/yolo/dota800/DOTA/ImageSets/Main/" + "train" +'.txt', index=False, header=None)
val.to_csv("/home/dingjin/Yolov3_DOTA/yolo/dota800/DOTA/ImageSets/Main/" + "val" +'.txt', index=False, header=None)
test.to_csv("/home/dingjin/Yolov3_DOTA/yolo/dota800/DOTA/ImageSets/Main/" + "test" +'.txt', index=False, header=None)


'''
800
in total 30019
train       18019
val         24019
test        30019
'''

'''
1024
in total    19219
train       11533
val         15376
test        19219
'''
