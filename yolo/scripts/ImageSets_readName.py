import os

def ListFilesToTxt(dir, file, wildcard, recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname = os.path.join(dir, name)
        if (os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname, file, wildcard, recursion)
        else:
            for ext in exts:
                if (name.endswith(ext)):
                    (filename, extension) = os.path.splitext(name)
                    file.write(filename + "\n")
                    break


def Test():
    dir = "/home/dingjin/Yolov3_DOTA/yolo/dota800/DOTA/Annotations/"                            # Annotations file
    outfile = "/home/dingjin/Yolov3_DOTA/yolo/dota800/DOTA/ImageSets/Main/all.txt"              # produce Name list
    wildcard = ".xml .jpg"

    file = open(outfile, "w")
    if not file:
        print("cannot open the file %s for writing" % outfile)
    ListFilesToTxt(dir, file, wildcard, 1)

    file.close()


Test()
