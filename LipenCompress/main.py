import os
import time
from PIL import Image , ImageOps

AUTHOR_TAG = 'atm' #SELECT YOUR OWN TAG  - MAX 3 DIGIT
IN_IMAGES_PATH = "inpictures/"
OUT_IMAGES_PATH = "outpictures/"
RESIZE_SIZE = (432,432)


def main():
    #Check if imput is vaild
    if AUTHOR_TAG == '' :
        print("Dodaj tag autora")
        return 1
    if not os.path.isdir(IN_IMAGES_PATH):
        print("Nie znaleziono filderu ze zdjęciami")
        print("Utwórz folder \"" + IN_IMAGES_PATH[:-1]+ "\"")
        return 1
    #Create out dir
    if not os.path.isdir(OUT_IMAGES_PATH):
        os.mkdir(str(OUT_IMAGES_PATH))
    #Get Images:
    image_list = getFiles(IN_IMAGES_PATH)
    image_amount = len(image_list)
    if image_amount == 0:
        print("Nie znaleziono żadnych zdjęć ")
        return 1
    #Perform Computations:
    for image in image_list:
        ti_m = os.path.getmtime(IN_IMAGES_PATH + image)
        pimage = Image.open(IN_IMAGES_PATH + image)
        pimage = ImageOps.exif_transpose(pimage)
        #add author sufix
        image = image[::-1].split('.')[-1][::-1] + "_" + str(AUTHOR_TAG) + "."+ image[::-1].split('.')[0][::-1]
        if pimage.size[0] > RESIZE_SIZE[0] and pimage.size[1] > RESIZE_SIZE[1]:
            #Resize image
            pimage = pimage.resize(RESIZE_SIZE)
            pimage.save(OUT_IMAGES_PATH + image)
        os.utime(OUT_IMAGES_PATH + image,(ti_m,ti_m))
    print("Added author tag")
    print("Changed images resolution")



def ListImages():
    image_list = []
    for file in os.listdir(IN_IMAGES_PATH):
        if os.path.isfile(os.path.join(IN_IMAGES_PATH, file)) and file.lower().endswith((".jpg", "jpeg", "png")):
            image_list.append(file)
    return image_list


def getFiles(dir):
    image_list = []
    if dir[-1] != '/': dir = dir + '/'

    if dir != IN_IMAGES_PATH and dir != IN_IMAGES_PATH[-1]:
        if OUT_IMAGES_PATH[-1] == '/':
            if not os.path.isdir(OUT_IMAGES_PATH + dir):
                if dir.split('/')[0] == IN_IMAGES_PATH or dir.split('/')[0] + '/' == IN_IMAGES_PATH:
                    sdir = "/".join(dir.split('/')[1:])  # do not save image path to image name
                os.mkdir(OUT_IMAGES_PATH + sdir)
        else:
            if not os.path.isdir(OUT_IMAGES_PATH + '/' + dir):
                if dir.split('/')[0] == IN_IMAGES_PATH or dir.split('/')[0] + '/' == IN_IMAGES_PATH:
                    sdir = "/".join(dir.split('/')[1:])  # do not save image path to image name
                os.mkdir(OUT_IMAGES_PATH + sdir)

    for file in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, file)):
            image_list.extend(getFiles(dir + file))
        if os.path.isfile(os.path.join(dir, file)) and file.lower().endswith((".jpg", "jpeg", "png")):
            if dir.split('/')[0] == IN_IMAGES_PATH or dir.split('/')[0]+'/' == IN_IMAGES_PATH:
                sdir = "/".join(dir.split('/')[1:]) # do not save image path to image name
            image_list.append(str(sdir + file))
    return image_list






if __name__ == '__main__':
    main()