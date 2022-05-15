from os import listdir
from os.path import isdir,isfile, join


def getImageFiles(dir,path):
    image_list = []
    if dir[-1] != '/': dir = dir + '/'

    for file in listdir(dir):
        if isdir(join(dir, file)):
            image_list.extend(getImageFiles(dir + file,path))
        if isfile(join(dir, file)) and file.lower().endswith((".jpg", "jpeg", "png")):
            image_list.append(str(dir + file))

    return image_list
