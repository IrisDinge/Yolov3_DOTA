import pandas as pd

data = pd.read_table("/home/dingjin/Yolov3_DOTA/yolo/results/Yolov3_800/merge/all.txt",
                     sep=' ', names=["image_name", "id", "score", "left", "top", "right", "bottom"])
new = data.groupby(data['image_name'])

for name, group in new:
    group.to_csv("/home/dingjin/Yolov3_DOTA/yolo/results/Yolov3_800/order/" + name + ".txt",
                 sep=' ', header=None, index=None, columns=['id', 'score', 'left', 'top', 'right', 'bottom'])
