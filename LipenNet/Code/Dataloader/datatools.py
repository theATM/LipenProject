from os import listdir
from os.path import isdir,isfile, join


def getImageFiles(dir,path):
    image_list = []
    if dir[-1] != '/': dir = dir + '/'

    for file in listdir(dir):
        if isdir(join(dir, file)):
            image_list.extend(getFiles(dir + file,path))
        if isfile(join(dir, file)) and file.lower().endswith((".jpg", "jpeg", "png")):
            if dir.split('/')[0]+'/' == path:
                sdir = "/".join(dir.split('/')[1:]) # do not save image path to image name
            image_list.append(str(sdir + file))

    return image_list
