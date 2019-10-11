import shutil
'''

create new folder <testGT>
read test.txt in ImageSets
copy labels from DOTA datasets

Elabels should be like:
<id> <xmin> <ymin> <xmax> <ymax>
<id> <xmin> <ymin> <xmax> <ymax>
<id> <xmin> <ymin> <xmax> <ymax>
...

'''

for line in open("/home/dingjin/Yolov3_DOTA/yolo/dota1024/DOTA/ImageSets/Main/test.txt"):
    img_name = line.strip('\n')
    file_name = "/home/dingjin/DOTA/train1024/Elabels/" + img_name + ".txt" or "/home/dingjin/DOTA/val1024/Elabels/" + img_name + ".txt"
    test_file = "/home/dingjin/Yolov3_DOTA/yolo/results/Yolov3/testGT/" + img_name + ".txt"
    print(test_file)
    shutil.copyfile(file_name, test_file)
