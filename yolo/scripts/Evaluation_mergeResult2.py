import os
import os.path
'''

write content of each .txt file in one .txt file <all.txt>  

'''


def mergeTxt(filepath, outfile):
    k = open(filepath+outfile, 'a+')
    for parent, dirnames, filenames in os.walk(filepath):
        for filepath in filenames:
            txtPath = os.path.join(parent,filepath)
            f = open(txtPath)
            k.write(f.read())
        k.close()
        print('finished')


if __name__ == "__main__":
    mergeTxt('/home/dingjin/Yolov3_DOTA/yolo/results/Yolov3_800/merge/', 'all.txt')
