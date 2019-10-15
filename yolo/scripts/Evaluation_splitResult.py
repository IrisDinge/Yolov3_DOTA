import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
original_txt = '/home/dingjin/Yolov3_DOTA/yolo/results/Yolov3_1024plane/comp4_det_test_plane.txt'
DateFrame = pd.read_table(original_txt, delimiter=' ', header=None, sep=' ',
                          names=["image_name", "score", "left", "top", "right", "bottom"])
dir_name = os.path.split(original_txt)[0]
name = os.path.split(original_txt)[1]
name = name.split('_', 3)[3]
id = name.split('.', 1)[0]
print(id)
print(dir_name + '/' + id + '/' + id + '.txt')
print(dir_name + '/' + id + '/' + 'results.txt')
print(dir_name + '/' + id + '/' + 'image_name' + '.txt')


class evluation():

    def __init__(self):

        df = DateFrame.drop(labels='score', axis=1)
        df = df.to_csv(dir_name + '/' + id + '/' + id + '.txt', sep=' ', header=None, index=False)
        all = pd.read_table(dir_name + '/' + id + '/' + id + '.txt',
                            delimiter=' ', header=None, sep=' ', names=["image_name", "left", "top", "right", "bottom"])
        image_name = all['image_name']

        dd = all.drop(labels='image_name', axis=1)
        dd = dd.to_csv(dir_name + '/' + id + '/' + 'results.txt', sep=' ', header=None, index=False)

        img = []
        for i in range(0, len(image_name)):
            img.append(image_name.iloc[i])
        such = []
        for i in img:
            index = [x for x in range(len(img)) if img[x] == i]
            such.append([i, index])
        final = list(such)

        m = pd.read_table(dir_name + '/' + id + '/' + 'results.txt', header=None, sep=' ',
                          names=["left", "top", "right", "bottom"])
        m.insert(0, 'class_name', id)
        # print(m)

        for i in range(len(final)):
            imn = final[i][0]
            index = final[i][1]
            for j in range(len(index)):
                p = index[0]
                q = index[j]
                n = m.loc[p:q]
                jieguo = n.to_csv(dir_name + '/' + id + '/' + imn + '.txt',
                                  header=None, sep=' ', index=None)
                print(jieguo)


Step = evluation()


