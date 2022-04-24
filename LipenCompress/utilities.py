import os


def getFiles(dir,in_path,out_path,createdir=False):
    image_list = []
    if dir[-1] != '/': dir = dir + '/'

    #Create subfolders
    if createdir is True and dir != in_path and dir != in_path[-1]:
        #Create new sub dir in out_path directory
        sdir = dir
        if dir.split('/')[0] + '/' == in_path:
            sdir = "/".join(dir.split('/')[1:])  # do not save image path to image name
        if not os.path.isdir(out_path + sdir):
            os.mkdir(out_path + sdir)

    for file in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, file)):
            image_list.extend(getFiles(dir + file,in_path,out_path,createdir))
        if os.path.isfile(os.path.join(dir, file)) and file.lower().endswith((".jpg", "jpeg", "png")):
            if dir.split('/')[0]+'/' == in_path:
                sdir = "/".join(dir.split('/')[1:]) # do not save image path to image name
            image_list.append(str(sdir + file))

    return image_list



def binary_decomposition(x):
    p = 2 ** (int(x).bit_length() - 1)
    while int(p):
        if p & int(x):
            yield p
        p //= 2